import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn

from train_args import parser, print_args

from time import time
from utils import * 
from models import *
from trainer import *


def main(args):
    save_folder = args.save_dir
    data_dir = args.data_root

    #log_folder = os.path.join(args.log_root, save_folder)
    log_folder = os.path.join(args.model_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, 'train', 'info')
    print_args(args, logger)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
    #if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    
    if args.model == "VGG16":
        net = vgg(dataset=args.dataset, depth=16)
        if args.mask:
            net = masked_vgg(dataset=args.dataset, depth=16)
    elif args.model == "VGG16-16":
        net = vgg_addlinear(dataset=args.dataset, depth=16)
        if args.mask:
            net = masked_vgg_addlinear(dataset=args.dataset, depth=16)
    elif args.model == "VGG19":
        net = vgg(dataset=args.dataset, depth=19)
        if args.mask:
            net = masked_vgg(dataset=args.dataset, depth=19)
    elif args.model == "ResNet18":
        net = ResNet18(dataset=args.dataset)
        if args.mask:
            net = masked_ResNet18(dataset=args.dataset)
    elif args.model == "ResNet50":
        net = ResNet50(dataset=args.dataset)
        if args.mask:
            net = masked_ResNet50(dataset=args.dataset)
    elif args.model == "vit" and args.dataset == "cifar10":
        net = vit_cifar_patch4_32()
        if args.mask:
            net = masked_vit_cifar_patch4_32()
    elif args.model == "vit" and args.dataset == "tiny-imagenet":
        net = vit_tiny_patch16_64()
        if args.mask:
            net = masked_vit_tiny_patch16_64()

    if args.load_ck:
        net.load_state_dict(
            torch.load(args.load_ck, map_location=lambda storage, loc: storage), strict=False)

    net.to(device)
    
    trainer = Trainer(args, logger)
    
    loss = nn.CrossEntropyLoss()
 
    
    kwargs = {'num_workers': 3, 'pin_memory': True} #if torch.cuda.is_available() else {}
    if args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(data_dir, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(data_dir, train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    else:
        from tiny_imagenet_dataset import TinyImageNet
        dataset_train = TinyImageNet(args.data_root, split='train', transform=transforms.Compose([
                            #transforms.Pad(4),
                            #transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ]))
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)


        dataset_val = TinyImageNet(args.data_root, split='val', transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ]))
        test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, **kwargs)
        
    
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    trainer.train(net, loss, device, train_loader, test_loader, optimizer=optimizer, scheduler=scheduler)
    


if __name__ == '__main__':
    args = parser()
    #print_args(args)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
