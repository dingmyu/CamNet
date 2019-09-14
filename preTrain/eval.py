import os
import cv2
import time
import logging
from argparse import ArgumentParser
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F

import segdata as datasets
import segtransforms as transforms
# from pspnet import PSPNet
from utils import AverageMeter, intersectionAndUnion, check_makedirs, colorize
cv2.ocl.setUseOpenCL(False)


# Setup
def get_parser():
    parser = ArgumentParser(description='PyTorch Semantic Segmentation Evaluation')
    parser.add_argument('--data_root', type=str, default='/mnt/sda1/hszhao/dataset/VOC2012', help='data root')
    parser.add_argument('--val_list1', type=str, default='/mnt/sda1/hszhao/dataset/VOC2012/list/val.txt', help='val list')
    parser.add_argument('--split', type=str, default='test', help='split in [train, val and test]')
    parser.add_argument('--backbone', type=str, default='resnet', help='backbone network type')
    parser.add_argument('--net_type', type=int, default=0, help='0-single branch, 1-div4 branch')
    parser.add_argument('--layers', type=int, default=50, help='layers number of based resnet')
    parser.add_argument('--classes', type=int, default=21, help='number of classes')
    parser.add_argument('--crop_h', type=int, default=473, help='validation crop size h')
    parser.add_argument('--crop_w', type=int, default=473, help='validation crop size w')
    parser.add_argument('--zoom_factor', type=int, default=1, help='zoom factor in final prediction map')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label in ground truth')
    parser.add_argument('--scales', type=float, default=[1.0], nargs='+', help='evaluation scales')
    parser.add_argument('--has_prediction', type=int, default=0, help='has prediction already or not')
    parser.add_argument('--batch_size', type=int, default=64*8, help='batch size')
    parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='used gpu')
    parser.add_argument('--workers', type=int, default=1, help='data loader workers')
    parser.add_argument('--model_path', type=str, default='exp/voc2012/psp50/model/train_epoch_100.pth', help='evaluation model path')
    parser.add_argument('--save_folder', type=str, default='exp/voc2012/psp50/result/epoch_100/val/ss', help='results save folder')
    parser.add_argument('--colors_path', type=str, default='data/voc2012/voc2012colors.txt', help='path of dataset colors')
    parser.add_argument('--names_path', type=str, default='data/voc2012/voc2012names.txt', help='path of dataset category names')
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
    global args, logger
    args = get_parser().parse_args()
    logger = get_logger()
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)
    logger.info(args)
    logger.info("=> creating model ...")
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]


    val_transform = transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.Crop([args.crop_h, args.crop_w], crop_type='center', padding=mean, ignore_label=args.ignore_label),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    val_data1 = datasets.SegData(split=args.split, data_root=args.data_root, data_list=args.val_list1, transform=val_transform)
    val_loader1 = torch.utils.data.DataLoader(val_data1, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    model_ppm = PPM().cuda()
    model_ppm = torch.nn.DataParallel(model_ppm)

    from pspnet import PSPNet
    model = PSPNet(backbone = args.backbone, layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, use_softmax=True, pretrained=False, syncbn=False).cuda()
    logger.info(model)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.enabled = True
    cudnn.benchmark = True
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
        
        

    checkpoint_ppm = torch.load(args.model_path.replace('.pth', '_ppm.pth'))
    model_ppm.load_state_dict(checkpoint_ppm['state_dict'], strict=False)
    
    cv2.setNumThreads(0)
    

    validate(val_loader1, val_data1.data_list, model, model_ppm)

    

def validate(val_loader, data_list, model, model_ppm):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    model_ppm.eval()
    end = time.time()
    import struct
    fp = open('filter_pose/result.bin', 'wb')
    for i, (input1, input2, translation, quaternions) in enumerate(val_loader):
        data_time.update(time.time() - end)
        
        
        input1 = input1.cuda(async=True)
        input_var1 = torch.autograd.Variable(input1)
        input2 = input2.cuda(async=True)
        input_var2 = torch.autograd.Variable(input2)
        x1_ICR, x1_PFR, x1_PRP = model(input_var1)
        x2_ICR, x2_PFR, x2_PRP = model(input_var2)
        print(x1_ICR.size(), x2_ICR.size())
        
        x1_ICR = (x1_ICR + x1_PFR + x1_PRP)/3
        x2_ICR = (x2_ICR + x2_PFR + x2_PRP)/3
        trans, quat = model_ppm(torch.cat([x1_ICR,x2_ICR], 1))
        
        result = torch.cat([trans,quat], 1).squeeze(3).squeeze(2).cpu().data.numpy()
        
        print(result[0])
        for line in result:
            for item in line:
                a = struct.pack('f', item)
                fp.write(a)
        
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 10 == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(val_loader),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
    fp.close()
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')





if __name__ == '__main__':
    main()
