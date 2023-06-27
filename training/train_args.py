import argparse

def parser():
    parser = argparse.ArgumentParser(description='Cifar')
    parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100 or imagenet)')
    parser.add_argument('--model', choices=['ResNet18', 'VGG16', 'WideResNet', 'ResNet20',
       'ResNet34', 'VGG19', 'VGG13', 'tim_VGG16', 'tim_ResNet18', 'nmt', 'vit', 'tim_vit','sparse_vit','sparse_tim_vit','ResNet50','tim_ResNet50','sparse_vgg16','VGG16-16'], default='VGG16',
        help='Which model to use')
    parser.add_argument('--data_root', default='/home/data/cifar10',
        help='the directory to save the dataset')
    parser.add_argument('--log_root', default='./checkpoint/log',
        help='the directory to save the logs or other imformations (e.g. images)')
    parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
    parser.add_argument('--load_ck', default=None, type=str)
    parser.add_argument('--save_dir', default='', help='the affix for the save folder')
    parser.add_argument('--milestones', type=int, default=[60,80], nargs="*")


    ## Training realted 
    parser.add_argument('--mask', action='store_true', help='whether to use masked model')
    parser.add_argument('--seed', type=int, default=1, help='The random seed')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size')
    parser.add_argument('--max_epoch', '-m_e', type=int, default=160, 
        help='the maximum numbers of the model see a sample')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum(defalt: 0.9)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="SGD weight decay(defalt: 1e-4)")
    parser.add_argument('--print_step', type=int, default=100, 
        help='print steps')
    parser.add_argument('--gpu', '-g', type=int, default=None, help='which gpu to use')
    parser.add_argument('--final_lambda', type=float, default=1e-6, help="penalty coefficient")
    parser.add_argument('--lamda_min', type=float, default=50, help="lamda min")
    parser.add_argument('--final_threshold', type=float, default=0.5, help="model final ratio")
   
    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))
