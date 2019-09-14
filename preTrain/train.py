import os
import time
import logging
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils2 import create_logger, AverageMeter, accuracy, save_checkpoint, load_state, IterLRScheduler, DistributedGivenIterationSampler, simple_group_split
from torchE.D import dist_init, average_gradients, DistModule
from tensorboardX import SummaryWriter

import segdata as datasets
import segtransforms as transforms
# from pspnet import PSPNet
from utils import AverageMeter, poly_learning_rate


# Setup
def get_parser():
    parser = ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--data_root', type=str, default='/mnt/sda1/hszhao/dataset/VOC2012', help='data root')
    parser.add_argument('--train_list', type=str, default='/mnt/sda1/hszhao/dataset/VOC2012/list/train.txt', help='train list')
    parser.add_argument('--val_list', type=str, default='/mnt/sda1/hszhao/dataset/VOC2012/list/val.txt', help='val list')
    parser.add_argument('--backbone', type=str, default='resnet', help='backbone network type')
    parser.add_argument('--net_type', type=int, default=0, help='0-single branch, 1-div4 branch')
    parser.add_argument('--layers', type=int, default=50, help='layers number of based resnet')
    parser.add_argument('--syncbn', type=int, default=1, help='adopt syncbn or not')
    parser.add_argument('--classes', type=int, default=21, help='number of classes')
    parser.add_argument('--crop_h', type=int, default=473, help='train crop size h')
    parser.add_argument('--crop_w', type=int, default=473, help='train crop size w')
    parser.add_argument('--scale_min', type=float, default=0.5, help='minimum random scale')
    parser.add_argument('--scale_max', type=float, default=2.0, help='maximum random scale')
    parser.add_argument('--rotate_min', type=float, default=-10, help='minimum random rotate')
    parser.add_argument('--rotate_max', type=float, default=10, help='maximum random rotate')
    parser.add_argument('--zoom_factor', type=int, default=1, help='zoom factor in final prediction map')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label in ground truth')
    parser.add_argument('--aux_weight', type=float, default=0.4, help='loss weight for aux branch')
    parser.add_argument('--use_aux', type=int, default=0, help='aux branch')
    parser.add_argument('--gpu', type=int, default=[0, 1, 2, 3], nargs='+', help='used gpu')
    parser.add_argument('--workers', type=int, default=4, help='data loader workers')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--bn_group', type=int, default=16, help='group number for sync bn')
    parser.add_argument('--batch_size_val', type=int, default=16, help='batch size for validation during training, memory and speed tradeoff')
    parser.add_argument('--base_lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--power', type=float, default=0.9, help='power in poly learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency (default: 10)')
    parser.add_argument('--save_step', type=int, default=10, help='model save step (default: 10)')
    parser.add_argument('--save_path', type=str, default='tmp', help='model and summary save path')
    parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint (default: none)')
    parser.add_argument('--weight', type=str, default='', help='path to weight (default: none)')
    parser.add_argument('--evaluate', type=int, default=0, help='evaluate on validation set, extra gpu memory needed and small batch_size_val is recommend')
    parser.add_argument('--port', default='23456', type=str)
    return parser


# logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def load_state(path, model,model_ppm, optimizer=None):
    def map_func(storage, location):
        return storage.cuda()
    if os.path.isfile(path):
        logger.info("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        checkpoint_ppm = torch.load(path.replace('.pth', '_ppm.pth'), map_location=map_func)
        args.start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        model_ppm.load_state_dict(checkpoint_ppm['state_dict'])
        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(path))


        
        
        
class PPM(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(PPM, self).__init__()
        self.cls_trans = nn.Sequential(
            nn.Conv2d(512*2, 512, kernel_size=1),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(p=dropout),
            nn.Conv2d(512, 3, kernel_size=1)
        )

        self.cls_quat = nn.Sequential(
            nn.Conv2d(512*2, 512, kernel_size=1),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(p=dropout),
            nn.Conv2d(512, 4, kernel_size=1)
        )
        
    def forward(self, x):
        out_trans = self.cls_trans(x)
        out_quat = self.cls_quat(x)
        return out_trans, out_quat

        
        
def main():
    global args, logger, writer
    args = get_parser().parse_args()
    import multiprocessing as mp
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    rank, world_size = dist_init(args.port)
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    #os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)
    #if len(args.gpu) == 1:
    #   args.syncbn = False
    if rank == 0:
        logger.info(args)

    if args.bn_group == 1:
        args.bn_group_comm = None
    else:
        assert world_size % args.bn_group == 0
        args.bn_group_comm = simple_group_split(world_size, rank, world_size // args.bn_group)

    if rank == 0:
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))

    from pspnet import PSPNet
    model = PSPNet(backbone=args.backbone, layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, syncbn=args.syncbn, group_size=args.bn_group, group=args.bn_group_comm).cuda()
    logger.info(model)
    model_ppm = PPM().cuda()
    # optimizer = torch.optim.SGD(model.parameters(), args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # newly introduced layer with lr x10
    optimizer = torch.optim.SGD(
        [{'params': model.layer0.parameters()},
         {'params': model.layer1.parameters()},
         {'params': model.layer2.parameters()},
         {'params': model.layer3.parameters()},
         {'params': model.layer4_ICR.parameters()},
         {'params': model.layer4_PFR.parameters()},
         {'params': model.layer4_PRP.parameters()},
         {'params': model_ppm.cls_trans.parameters(), 'lr': args.base_lr * 10},
         {'params': model_ppm.cls_quat.parameters(), 'lr': args.base_lr * 10}
        ],
        lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #model = torch.nn.DataParallel(model).cuda()
    model = DistModule(model)
    model_ppm = DistModule(model_ppm)
    cudnn.enabled = True
    cudnn.benchmark = True
    criterion = nn.L1Loss().cuda()

    if args.weight:
        def map_func(storage, location):
            return storage.cuda()
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location=map_func)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        load_state(args.resume, model, model_ppm, optimizer)

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transforms.Compose([
        transforms.Resize(size=(256,256)),
        #transforms.RandomGaussianBlur(),
        transforms.Crop([args.crop_h, args.crop_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
        transforms.ColorJitter([0.4,0.4,0.4]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    
    train_data = datasets.SegData(split='train', data_root=args.data_root, data_list=args.train_list, transform=train_transform)
    train_sampler = DistributedSampler(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    if args.evaluate:
        val_transform = transforms.Compose([
            transforms.Resize(size=(256,256)),
            transforms.Crop([args.crop_h, args.crop_w], crop_type='center', padding=mean, ignore_label=args.ignore_label),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        val_data = datasets.SegData(split='val', data_root=args.data_root, data_list=args.val_list, transform=val_transform)
        val_sampler = DistributedSampler(val_data)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    for epoch in range(args.start_epoch, args.epochs + 1):
        t_loss_train, r_loss_train= train(train_loader, model, model_ppm, criterion, optimizer, epoch, args.zoom_factor, args.batch_size, args.aux_weight)
        if rank == 0:
            writer.add_scalar('t_loss_train', t_loss_train, epoch)
            writer.add_scalar('r_loss_train', r_loss_train, epoch)
        # write parameters histogram costs lots of time
        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param, epoch)

        if epoch % args.save_step == 0 and rank == 0:
            filename = args.save_path + '/train_epoch_' + str(epoch) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            filename_ppm = args.save_path + '/train_epoch_' + str(epoch) + '_ppm.pth'
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            torch.save({'epoch': epoch, 'state_dict': model_ppm.state_dict(), 'optimizer': optimizer.state_dict()}, filename_ppm)
            #if epoch / args.save_step > 2:
            #    deletename = args.save_path + '/train_epoch_' + str(epoch - args.save_step*2) + '.pth'
            #    os.remove(deletename)
        if args.evaluate:
            t_loss_val, r_loss_val= validate(val_loader, model, model_ppm, criterion)
            writer.add_scalar('t_loss_val', t_loss_val, epoch)
            writer.add_scalar('r_loss_val', r_loss_val, epoch)
    writer.close()


def train(train_loader, model, model_ppm, criterion, optimizer, epoch, zoom_factor, batch_size, aux_weight):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    r_loss_meter = AverageMeter()
    t_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    


    model.train()
    model_ppm.train()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    #print(rank)
    end = time.time()
    for i, (input1, input2, translation, quaternions) in enumerate(train_loader):
        # to avoid bn problem in ppm module with bin size 1x1, sometimes n may get 1 on one gpu during the last batch, so just discard
        # if input.shape[0] < batch_size:
        #     continue
        data_time.update(time.time() - end)
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = args.epochs * len(train_loader)
        index_split = 4
        poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power, index_split=index_split)

        input1 = input1.cuda()
        input_var1 = torch.autograd.Variable(input1)
        input2 = input2.cuda()
        input_var2 = torch.autograd.Variable(input2)
        x1_ICR, x1_PFR, x1_PRP = model(input_var1)
        x2_ICR, x2_PFR, x2_PRP = model(input_var2)
 
        x1_ICR = (x1_ICR + x1_PFR + x1_PRP)/3
        x2_ICR = (x2_ICR + x2_PFR + x2_PRP)/3
        trans, quat = model_ppm(torch.cat([x1_ICR,x2_ICR], 1))

        
        translation = translation.float().cuda(async=True)
        translation_var = torch.autograd.Variable(translation)
        quaternions = quaternions.float().cuda(async=True)
        quaternions_var = torch.autograd.Variable(quaternions)
        
        t_loss = criterion(trans, translation_var) / world_size
        r_loss = criterion(quat, quaternions_var) / world_size
        loss = r_loss + t_loss

        optimizer.zero_grad()
        loss.backward()
        average_gradients(model)
        optimizer.step()

        reduced_loss = loss.data.clone()
        reduced_t_loss = t_loss.data.clone()
        reduced_r_loss = r_loss.data.clone()
        dist.all_reduce(reduced_loss)
        dist.all_reduce(reduced_t_loss)
        dist.all_reduce(reduced_r_loss)

        r_loss_meter.update(reduced_r_loss[0], input1.size(0))
        t_loss_meter.update(reduced_t_loss[0], input1.size(0))
        loss_meter.update(reduced_loss[0], input1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if rank == 0:
            if (i + 1) % args.print_freq == 0:
                logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'rLoss {r_loss_meter.val:.4f} '
                            'tLoss {t_loss_meter.val:.4f} '
                            'Loss {loss_meter.val:.4f} '.format(epoch, args.epochs, i + 1, len(train_loader),
                                                              batch_time=batch_time,
                                                              data_time=data_time,
                                                              remain_time=remain_time,
                                                              t_loss_meter=t_loss_meter,
                                                              r_loss_meter=r_loss_meter,
                                                              loss_meter=loss_meter))
            writer.add_scalar('loss_train_batch_r', r_loss_meter.val, current_iter)
            writer.add_scalar('loss_train_batch_t', t_loss_meter.val, current_iter)


    if rank == 0:
        logger.info('Train result at epoch [{}/{}].'.format(epoch, args.epochs))
    return t_loss_meter.avg, r_loss_meter.avg


def validate(val_loader, model, model_ppm, criterion):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank == 0:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    r_loss_meter = AverageMeter()
    t_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    


    model.eval()
    model_ppm.eval()
    
    end = time.time()
    
    for i, (input1, input2, translation, quaternions) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input1 = input1.cuda()
        input_var1 = torch.autograd.Variable(input1)
        input2 = input2.cuda()
        input_var2 = torch.autograd.Variable(input2)
        output1= model(input_var1)
        output2 = model(input_var2)

        trans, quat = model_ppm(torch.cat([output1,output2], 1))
        
        translation = translation.float().cuda(async=True)
        translation_var = torch.autograd.Variable(translation)
        quaternions = quaternions.float().cuda(async=True)
        quaternions_var = torch.autograd.Variable(quaternions)
        
        t_loss = criterion(trans, translation_var) / world_size
        r_loss =  criterion(quat, quaternions_var) / world_size
        loss = t_loss +r_loss
        reduced_loss = loss.data.clone()
        reduced_t_loss = t_loss.data.clone()
        reduced_r_loss = r_loss.data.clone()
        dist.all_reduce(reduced_loss)
        dist.all_reduce(reduced_t_loss)
        dist.all_reduce(reduced_r_loss)
        t_loss_meter.update(reduced_t_loss[0], input1.size(0))
        r_loss_meter.update(reduced_r_loss[0], input1.size(0))
        loss_meter.update(reduced_loss[0], input1.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()


        if (i + 1) % 10 == 0 and rank == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'tLoss {t_loss_meter.val:.4f} ({t_loss_meter.avg:.4f}) '
                        'rLoss {r_loss_meter.val:.4f} ({r_loss_meter.avg:.4f}) '.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          t_loss_meter=t_loss_meter,
                                                          r_loss_meter=r_loss_meter))
    if rank == 0:
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return t_loss_meter.avg, r_loss_meter.avg


if __name__ == '__main__':
    main()

