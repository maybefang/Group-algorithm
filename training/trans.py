import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn

# from train_args import parser, print_args

from time import time
from utils import * 
from models import *
from trainer import *


def main():

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # net = vgg(dataset='tiny-imagenet', depth=16)
    # dirnet = masked_vgg(dataset='tiny-imagenet', depth=16)
    # load_path = './checkpoint/vgg16_tim_nop_lr0.2_60120/best_acc_model.pth' #./pretrain/guanfang_trans_vgg16.pth

    # net = vgg(dataset='cifar10', depth=16)
    # dirnet = masked_vgg(dataset='cifar10', depth=16)
    # load_path = './checkpoint/vgg16_cifar10_nop_lr0.1_60120/best_acc_model.pth' #./pretrain/guanfang_cifar10_vgg16.pth
    
    # net = vgg_addlinear(dataset='cifar100', depth=16)
    # dirnet = masked_vgg_addlinear(dataset='cifar10', depth=16)
    # load_path = './checkpoint/vgg16-16_cifar100_nop_lr0.1_60/best_acc_model.pth' #./pretrain/guanfang_cifar10_vgg16.pth

    net = vgg(dataset='cifar100', depth=16)
    dirnet = masked_vgg(dataset='cifar100', depth=16)
    load_path = './checkpoint/vgg16_cifar100_nop_lr0.1_60120/best_acc_model.pth'

    # net = ResNet18(dataset='tiny-imagenet')
    # dirnet = masked_ResNet18(dataset='tiny-imagenet')
    # load_path = './checkpoint/tim_resnet18_nop_lr0.1_80170/best_acc_model.pth'

    # net = ResNet18(dataset='cifar100')
    # dirnet = masked_ResNet18(dataset='cifar100')
    # load_path = './checkpoint/resnet18_cifar100_nop_lr0.1_60120/best_acc_model.pth'

    # net = ResNet50(dataset='tiny-imagenet')
    # dirnet = masked_ResNet50(dataset='tiny-imagenet')
    # load_path = './checkpoint/resnet50_nop_lr0.2_60120/best_acc_model.pth'

    # net = vgg(dataset='tiny-imagenet', depth=19)
    # dirnet = masked_vgg(dataset='tiny-imagenet', depth=19)
    # load_path = './checkpoint/vgg19_nop_lr0.1/best_acc_model.pth'
    
    
    net.load_state_dict(
            torch.load(load_path, map_location=lambda storage, loc: storage), strict=False)

    # net.to(device)
    
    weights_bns=[]

    weights=[]

    for l in net.modules():#named_parameters():#modules():
        #print(l)

        if isinstance(l,nn.Conv2d):# or isinstance(l,nn.Linear):
            #pl=[l[0] for l in l.named_parameters()]
            #print(pl)
            weights_bns.append(l.weight)
            #weights_bns.append(1)
        if isinstance(l,nn.Linear) or isinstance(l,nn.BatchNorm2d) or isinstance(l,nn.BatchNorm1d):
            #pl=[l[0] for l in l.named_parameters()]
            #print("-------------------------",pl)
            weights_bns.append(l.weight)
            weights_bns.append(l.bias)
            #weights_bns.append(1)
        #weights_bns.append(l[1])

    print("++++++++++++++++++++++++++++++++++++")
    for l in net.named_parameters():
        weights.append(l[1])
        #weights.append(1)
        # print(l[0])

    print('weights_bns nums:',len(weights_bns))
    print('weights nums:',len(weights))

    #n = len(weights)
    #for i in range(n):
    #    print(weights_bns[i].equal(weights[i]))



    #for l in dirnet.named_parameters():
    #    print(l[1])
    #    break

    i=0
    for l in dirnet.modules():#named_parameters():#modules():

        if isinstance(l,nn.Conv2d):# or isinstance(l,nn.Linear):
            l.weight = weights[i]
            i += 1
        if isinstance(l,nn.Linear) or isinstance(l,nn.BatchNorm2d) or isinstance(l,nn.BatchNorm1d):
            l.weight = weights[i]
            i += 1
            l.bias = weights[i]
            i += 1
    print("i =",i)
    #for l in dirnet.named_parameters():
    #    print(weights_bns[0].equal(l[1]))
    #    break
    torch.save(dirnet.state_dict(), './pretrain/guanfang_cifar100_vgg16.pth')

    


if __name__ == '__main__':
    # args = parser()
    #print_args(args)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main()
