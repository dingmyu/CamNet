import torch
from torch import nn
import torch.nn.functional as F
# from torchvision import models
# import resnet as models
from torchE.nn import SyncBatchNorm2d
import torchvision
# from utils import init_weights




class PSPNet(nn.Module):
    def __init__(self, backbone='resnet', layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, use_softmax=True, pretrained=True, syncbn=True, group_size=8, group=None):
        super(PSPNet, self).__init__()
        assert layers in [18, 34, 50, 101, 152]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.use_softmax = use_softmax

        if backbone == 'resnet':
            import resnet as models
        else:
            raise NameError('Backbone type not defined!')

        if syncbn:
            # from lib.syncbn import SynchronizedBatchNorm2d as BatchNorm
            def BNFunc(*args, **kwargs):
                return SyncBatchNorm2d(*args, **kwargs, group_size=group_size, group=group, sync_stats=True)
            BatchNorm = BNFunc
        else:
            from torch.nn import BatchNorm2d as BatchNorm
        models.BatchNorm = BatchNorm

        if layers == 34:
            resnet = models.resnet34(pretrained=pretrained)
        elif layers == 18:
            resnet = models.resnet18(pretrained=pretrained)
        elif layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
            
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3 = resnet.layer1, resnet.layer2, resnet.layer3
        self.layer4_ICR, self.layer4_PFR, self.layer4_PRP = resnet.layer4_ICR, resnet.layer4_PFR, resnet.layer4_PRP
        self.avgpool = nn.AvgPool2d(7, stride=1)
            # init_weights(self.aux)
        # comment to use default initialization
        # init_weights(self.ppm)
        # init_weights(self.cls)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_ICR = self.layer4_ICR(x)
        x_PFR = self.layer4_PFR(x)
        x_PRP = self.layer4_PRP(x)

        x_ICR = self.avgpool(x_ICR)
        x_PFR = self.avgpool(x_PFR)
        x_PRP = self.avgpool(x_PRP)

        return x_ICR, x_PFR, x_PRP


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sim_data = torch.autograd.Variable(torch.rand(2, 3, 473, 473)).cuda(async=True)
    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, use_softmax=True, use_aux=True, pretrained=True, syncbn=True).cuda()
    print(model)
    output, _ = model(sim_data)
    print('PSPNet', output.size())
