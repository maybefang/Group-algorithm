import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from .masked_nn import MaskedConv2d, MaskedLinear


__all__ = ['masked_vgg','masked_vgg_addlinear']

defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class masked_vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(masked_vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg
        self.feature = self.make_layers(cfg, True)
        flag = 1

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny-imagenet':
            num_classes = 200
            flag = 4


        self.classifier = nn.Sequential(
              MaskedLinear(cfg[-1]*flag, 512),
            #   SparseLinear(cfg[-1], 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace=True),
              MaskedLinear(512, num_classes)
            #   SparseLinear(512, num_classes)
            )
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = MaskedConv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, MaskedConv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, MaskedLinear):
                m.reset_parameters()

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

class masked_vgg_addlinear(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(masked_vgg_addlinear, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg
        self.feature = self.make_layers(cfg, True)
        flag = 1

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny-imagenet':
            num_classes = 200
            flag = 4


        self.classifier = nn.Sequential(
              MaskedLinear(cfg[-1]*flag, 512),
            #   SparseLinear(cfg[-1], 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace=True),
              MaskedLinear(512, 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace=True),
              MaskedLinear(512, num_classes)
            #   SparseLinear(512, num_classes)
            )
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = MaskedConv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, MaskedConv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, MaskedLinear):
                m.reset_parameters()

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

