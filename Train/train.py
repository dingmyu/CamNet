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
import cv2

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


class ICR(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(ICR, self).__init__()
        self.cls_proj = nn.Sequential(
            nn.Conv2d(512*2, 512, kernel_size=1),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(p=dropout),
            nn.Conv2d(512, 3, kernel_size=1)
        )

        
    def forward(self, x):
        out_proj = self.cls_proj(x)
        return out_proj

        
        
class PFR(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(PFR, self).__init__()
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

class PRP(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(PRP, self).__init__()
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

def _pairwise_distance(x, y, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.
    cor_matx = torch.matmul(x, x.t())
    cor_maty = torch.matmul(y, y.t())
    cor_mat = torch.matmul(x, y.t())
    norm_matx = cor_matx.diag()
    norm_maty = cor_maty.diag()
    distances = norm_matx.unsqueeze(1) - 2 * cor_mat + norm_maty.unsqueeze(0)
    distances = F.relu(distances)#/512
    #print(distances,'dist')
    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)
    #print(distances,'dist_squared')
    return distances
    
    
class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """
    def __init__(self, margin=0.1, hardest=False, squared=False):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared

    def forward(self, x, y):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        pairwise_dist = _pairwise_distance(x, y, squared=self.squared)
        #print(pairwise_dist.size())
        batch_size = int(x.size()[0])

        if self.hardest:
            # Get the hardest positive pairs
            valid_positive_dist = pairwise_dist[:int(batch_size/3),:int(batch_size/3)]
            hardest_positive_dist, _ = torch.max(valid_positive_dist.diag(), dim=0, keepdim=True)
            
            valid_medium_dist = pairwise_dist[int(batch_size/3):int(batch_size/3*2),int(batch_size/3):int(batch_size/3*2)]
            hardest_medium_dist, _ = torch.max(valid_medium_dist.diag(), dim=0, keepdim=True)
            easiest_medium_dist, _ = torch.min(valid_medium_dist.diag(), dim=0, keepdim=True)

            # Get the hardest negative pairs
            anchor_negative_dist = pairwise_dist[int(batch_size/3*2):batch_size,int(batch_size/3*2):batch_size]
            hardest_negative_dist, _ = torch.min(anchor_negative_dist.diag(), dim=0, keepdim=True)

            # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
            triplet_loss = F.relu(hardest_positive_dist - easiest_medium_dist + self.margin) + F.relu(hardest_medium_dist - hardest_negative_dist + self.margin)
            triplet_loss = torch.mean(triplet_loss)
        else:
            valid_positive_dist = pairwise_dist[:int(batch_size/3),:int(batch_size/3)].diag()
            valid_medium_dist = pairwise_dist[int(batch_size/3):int(batch_size/3*2),int(batch_size/3):int(batch_size/3*2)].diag()
            anchor_negative_dist = pairwise_dist[int(batch_size/3*2):batch_size,int(batch_size/3*2):batch_size].diag()
            
            triplet_loss = F.relu(valid_positive_dist - valid_medium_dist + self.margin) + F.relu(valid_medium_dist - anchor_negative_dist + self.margin)

            # Remove negative losses (i.e. the easy triplets)
            triplet_loss = F.relu(triplet_loss)

            # Count number of hard triplets (where triplet_loss > 0)
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)

            triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss
    
    
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
    model_icr = ICR().cuda()
    model_pfr = PFR().cuda()
    model_prp = PRP().cuda()
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
         {'params': model_icr.cls_proj.parameters(), 'lr': args.base_lr * 10},
         {'params': model_pfr.cls_trans.parameters(), 'lr': args.base_lr * 10},
         {'params': model_pfr.cls_quat.parameters(), 'lr': args.base_lr * 10},
         {'params': model_prp.cls_trans.parameters(), 'lr': args.base_lr * 10},
         {'params': model_prp.cls_quat.parameters(), 'lr': args.base_lr * 10},
        ],
        lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #model = torch.nn.DataParallel(model).cuda()
    model = DistModule(model)
    model_icr = DistModule(model_icr)
    model_pfr = DistModule(model_pfr)
    model_prp = DistModule(model_prp)
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
            print('Loaded pre-trained model.')
            
            checkpoint = torch.load(args.weight.replace('.pth', '_ppm.pth'), map_location=map_func)
            model_pfr.load_state_dict(checkpoint['state_dict'])
            model_prp.load_state_dict(checkpoint['state_dict'])
            print('Loaded pre-trained model for sub models.')
            
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

#     if args.resume:
#         load_state(args.resume, model, model_ppm, optimizer)

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
    TripletLoss = HardTripletLoss(hardest=False, squared=False)

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
        t_loss_train, r_loss_train= train(train_loader, train_transform, model, model_icr, model_pfr, model_prp, TripletLoss, criterion, optimizer, epoch, args.zoom_factor, args.batch_size, args.aux_weight)
        if rank == 0:
            writer.add_scalar('t_loss_train', t_loss_train, epoch)
            writer.add_scalar('r_loss_train', r_loss_train, epoch)
        # write parameters histogram costs lots of time
        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param, epoch)

        if epoch % args.save_step == 0 and rank == 0:
            filename = args.save_path + '/train_epoch_' + str(epoch) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            filename_icr = args.save_path + '/train_epoch_' + str(epoch) + '_icr.pth'
            filename_pfr = args.save_path + '/train_epoch_' + str(epoch) + '_pfr.pth'
            filename_prp = args.save_path + '/train_epoch_' + str(epoch) + '_prp.pth'
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            torch.save({'epoch': epoch, 'state_dict': model_icr.state_dict(), 'optimizer': optimizer.state_dict()}, filename_icr)
            torch.save({'epoch': epoch, 'state_dict': model_pfr.state_dict(), 'optimizer': optimizer.state_dict()}, filename_pfr)
            torch.save({'epoch': epoch, 'state_dict': model_prp.state_dict(), 'optimizer': optimizer.state_dict()}, filename_prp)
            #if epoch / args.save_step > 2:
            #    deletename = args.save_path + '/train_epoch_' + str(epoch - args.save_step*2) + '.pth'
            #    os.remove(deletename)
#         if args.evaluate:
#             t_loss_val, r_loss_val= validate(val_loader, model, model_ppm, criterion)
#             writer.add_scalar('t_loss_val', t_loss_val, epoch)
#             writer.add_scalar('r_loss_val', r_loss_val, epoch)
    writer.close()


def get_coarse_quaternion(anchor_quat, relative_quat):
    s_anchor, v_anchor = anchor_quat[:,0:1], anchor_quat[:,1:]
    s_relative, v_relative = relative_quat[:,0:1], relative_quat[:,1:]
    s_new = s_anchor.mul(s_relative) - v_anchor.mul(v_relative).sum(1, keepdim=True)
    v_new = s_anchor.mul(v_relative) + s_relative.mul(v_anchor) + v_anchor.cross(v_relative, dim= 1)
    return torch.cat([s_new, v_new], 1)

def pose_distance(rt1, rt2):
    r1 = rt1[:,:4]
    t1 = rt1[:,4:]
    r2 = rt2[:,:4]
    t2 = rt2[:,4:]
    distances = 2*np.arccos(abs(np.sum(r1*r2, axis = 1)))*180/np.pi/40 + np.linalg.norm(t1-t2,axis = 1)
    distances[np.isnan(distances)] = 100
    return distances


def get_retrival_info(filename):
    f = open(filename)
    name_list = []
    rt_list = []
    for line in f.readlines():
        line = line.strip().split()
        name_list.append(line[0])
        rt_list.append([float(item) for item in line[1:]])
        
    return name_list, np.array(rt_list)

def train(train_loader, train_transform, model, model_icr, model_pfr, model_prp, TripletLoss, criterion, optimizer, epoch, zoom_factor, batch_size, aux_weight):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    r_loss_meter = AverageMeter()
    t_loss_meter = AverageMeter()
    proj_loss_meter = AverageMeter()
    triple_loss_meter = AverageMeter()
    rfine_loss_meter = AverageMeter()
    tfine_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    


    model.train()
    model_icr.train()
    model_pfr.train()
    model_prp.train()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    #print(rank)
    end = time.time()
    for index, (image_anchor, image1, image2, image3, relative_t1, relative_r1, relative_t2, relative_r2, relative_t3, relative_r3, image1_r, image2_r, image3_r, anchor_name, absolute_r1, absolute_t1, absolute_r2, absolute_t2, absolute_r3, absolute_t3, absolute_ranchor, absolute_tanchor) in enumerate(train_loader):
        # to avoid bn problem in ppm module with bin size 1x1, sometimes n may get 1 on one gpu during the last batch, so just discard
        # if input.shape[0] < batch_size:
        #     continue
        data_time.update(time.time() - end)
        current_iter = (epoch - 1) * len(train_loader) + index + 1
        max_iter = args.epochs * len(train_loader)
        index_split = 4
        poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power, index_split=index_split)

#         print(image_anchor.size())
        image_anchor = torch.cat([image_anchor, image_anchor, image_anchor],0)
        image1 = torch.cat([image1, image2, image3],0)
        relative_t1 = torch.cat([relative_t1, relative_t2, relative_t3],0)
        relative_r1 = torch.cat([relative_r1, relative_r2, relative_r3],0)
        image1_r = torch.cat([image1_r, image2_r, image3_r],0)
#         print(image_anchor.size())
        
        image_anchor = image_anchor.cuda()
        image_anchor_var = torch.autograd.Variable(image_anchor)
        image1 = image1.cuda()
        image1_var = torch.autograd.Variable(image1)
        x1_ICR, x1_PFR, x1_PRP = model(image_anchor_var)
        x2_ICR, x2_PFR, x2_PRP = model(image1_var)
        
        
        proj_ICR = model_icr(torch.cat([x1_ICR,x2_ICR], 1))
        trans, quat = model_pfr(torch.cat([x1_PFR,x2_PFR], 1))
        
        
        translation = relative_t1.float().cuda(async=True)
        translation_var = torch.autograd.Variable(translation)
        quaternions = relative_r1.float().cuda(async=True)
        quaternions_var = torch.autograd.Variable(quaternions)
        proj = image1_r.float().cuda(async=True)
        proj_var = torch.autograd.Variable(proj)
        
        triple_loss = TripletLoss(x1_ICR.squeeze(3).squeeze(2), x2_ICR.squeeze(3).squeeze(2)) / world_size
        t_loss = criterion(trans.squeeze(3).squeeze(2), translation_var) / world_size
        r_loss = criterion(quat.squeeze(3).squeeze(2), quaternions_var) / world_size
        proj_loss = criterion(proj_ICR.squeeze(3).squeeze(2), proj_var) / world_size
        
        #########################################################################################
        
        
        absolute_r1 = torch.cat([absolute_r1, absolute_r2, absolute_r3],0)
        absolute_t1 = torch.cat([absolute_t1, absolute_t2, absolute_t3],0)
        
        course_t = absolute_t1 + trans.squeeze(3).squeeze(2).data.cpu()
        course_r = get_coarse_quaternion(absolute_r1, quat.squeeze(3).squeeze(2).data.cpu())
        course_rt = torch.cat([course_r, course_t], 1).numpy()
        #print(anchor_name, absolute_r1.size(), quat.size(), course_t, course_r)
        
        name_list = []
        rt_list = []
        for item in anchor_name:
            name_info, rt_info = get_retrival_info(item.replace(args.data_root, '/mnt/lustre/dingmingyu/Research/ICCV19/CamNet/scripts/retrival_lists/'))
            name_list.append(name_info)
            rt_list.append(rt_info)
            
        #print(len(name_list), name_list)
        
        fine_list = []
        fine_rt_list = []
        r_fine_loss = 0
        t_fine_loss = 0
        for i in range(course_rt.shape[0]):
            distances = pose_distance(course_rt[i:i+1], rt_list[i % int(course_rt.shape[0]/3)])
            num = np.argmin(distances)
            #print(num, distances[num])
            fine_list.append(name_list[i % int(course_rt.shape[0]/3)][num])
            fine_rt_list.append(rt_list[i % int(course_rt.shape[0]/3)][num])
        
        for i in range(len(anchor_name)):
            #print(anchor_name[i], args.data_root + fine_list[i])
            fine_anchor, fine_1, fine_2, fine_3 = cv2.imread(anchor_name[i].replace('pose.txt', 'color.png')), cv2.imread(args.data_root + fine_list[i].replace('pose.txt', 'color.png')), cv2.imread(args.data_root + fine_list[i+len(anchor_name)].replace('pose.txt', 'color.png')), cv2.imread(args.data_root + fine_list[i+len(anchor_name)+len(anchor_name)].replace('pose.txt', 'color.png'))
            fine_anchor, fine_1, fine_2, fine_3 = train_transform(fine_anchor, fine_1, fine_2, fine_3)
            fine_anchor = torch.cat([fine_anchor.unsqueeze(0), fine_anchor.unsqueeze(0), fine_anchor.unsqueeze(0)], 0)
            fine_1 = torch.cat([fine_1.unsqueeze(0), fine_2.unsqueeze(0), fine_3.unsqueeze(0)], 0)
            
            fine_anchor_r = absolute_ranchor[i:i+1]
            fine_anchor_r = torch.cat([fine_anchor_r, fine_anchor_r, fine_anchor_r], 0)
            fine_anchor_t = absolute_tanchor[i:i+1]
            fine_anchor_t = torch.cat([fine_anchor_t, fine_anchor_t, fine_anchor_t], 0)
            fine_imgs_rt = np.array([fine_rt_list[i], fine_rt_list[i+len(anchor_name)], fine_rt_list[i+len(anchor_name)+len(anchor_name)]]).astype(np.float32)
            fine_imgs_r = torch.from_numpy(fine_imgs_rt[:,:4])
            fine_imgs_t = torch.from_numpy(fine_imgs_rt[:,4:])
            #print(fine_anchor_r.size(), fine_anchor_t.size(), fine_imgs_r.size(), fine_imgs_t.size())
            fine_rela_t = fine_imgs_t - fine_anchor_t
            fine_rela_t_var = torch.autograd.Variable(fine_rela_t.cuda())
            fine_anchor_r[:,1:] *= -1
            fine_rela_r = get_coarse_quaternion(fine_anchor_r, fine_imgs_r)
            fine_rela_r_var = torch.autograd.Variable(fine_rela_r.cuda())
            #print(fine_rela_r.size(), fine_rela_t.size(), fine_rela_r, fine_rela_t)
            fine_anchor_var = torch.autograd.Variable(fine_anchor.cuda())
            fine_imgs_var = torch.autograd.Variable(fine_1.cuda())
            _, _, anchor_PRP = model(fine_anchor_var)
            _, _, imgs_PRP = model(fine_imgs_var)
            trans_PRP, quat_PRP = model_prp(torch.cat([anchor_PRP,imgs_PRP], 1))
            r_fine_loss += criterion(quat_PRP.squeeze(3).squeeze(2), fine_rela_r_var) / world_size / len(anchor_name)
            t_fine_loss += criterion(trans_PRP.squeeze(3).squeeze(2), fine_rela_t_var) / world_size / len(anchor_name)
            if rank == 0:
                print(anchor_name[i], args.data_root + fine_list[i], fine_list[i+len(anchor_name)], fine_list[i+len(anchor_name)+len(anchor_name)], fine_anchor_r, fine_anchor_t, fine_rela_t, fine_rela_r)
            #print(anchor_name[i], args.data_root + fine_list[i], fine_anchor_r[i], fine_anchor_t[i], fine_rt_list[i], fine_rt_list[i+len(anchor_name)], fine_rt_list[i+len(anchor_name)+len(anchor_name)])

        loss = r_loss + t_loss + proj_loss + triple_loss + r_fine_loss + t_fine_loss

        optimizer.zero_grad()
        loss.backward()
        average_gradients(model)
        optimizer.step()

        reduced_loss = loss.data.clone()
        reduced_t_loss = t_loss.data.clone()
        reduced_r_loss = r_loss.data.clone()
        reduced_proj_loss = proj_loss.data.clone()
        reduced_triple_loss = triple_loss.data.clone()
        reduced_rfine_loss = r_fine_loss.data.clone()
        reduced_tfine_loss = t_fine_loss.data.clone()
        dist.all_reduce(reduced_loss)
        dist.all_reduce(reduced_t_loss)
        dist.all_reduce(reduced_r_loss)
        dist.all_reduce(reduced_proj_loss)
        dist.all_reduce(reduced_triple_loss)
        dist.all_reduce(reduced_rfine_loss)
        dist.all_reduce(reduced_tfine_loss)

        r_loss_meter.update(reduced_r_loss[0], image_anchor.size(0))
        t_loss_meter.update(reduced_t_loss[0], image_anchor.size(0))
        proj_loss_meter.update(reduced_proj_loss[0], image_anchor.size(0))
        triple_loss_meter.update(reduced_triple_loss[0], image_anchor.size(0))
        rfine_loss_meter.update(reduced_rfine_loss[0], image_anchor.size(0))
        tfine_loss_meter.update(reduced_tfine_loss[0], image_anchor.size(0))
        loss_meter.update(reduced_loss[0], image_anchor.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if rank == 0:
            if (index + 1) % args.print_freq == 0:
                logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'rLoss {r_loss_meter.val:.4f} '
                            'tLoss {t_loss_meter.val:.4f} '
                            'projLoss {proj_loss_meter.val:.4f} '
                            'tripleLoss {triple_loss_meter.val:.4f} '
                            'rfineLoss {rfine_loss_meter.val:.4f} '
                            'tfineLoss {tfine_loss_meter.val:.4f} '
                            'Loss {loss_meter.val:.4f} '.format(epoch, args.epochs, index + 1, len(train_loader),
                                                              batch_time=batch_time,
                                                              data_time=data_time,
                                                              remain_time=remain_time,
                                                              t_loss_meter=t_loss_meter,
                                                              r_loss_meter=r_loss_meter,
                                                              proj_loss_meter=proj_loss_meter,
                                                              triple_loss_meter=triple_loss_meter,
                                                              rfine_loss_meter=rfine_loss_meter,
                                                              tfine_loss_meter=tfine_loss_meter,
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

