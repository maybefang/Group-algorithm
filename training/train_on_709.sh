#!/bin/bash
#SBATCH -A lyj06 # 自己所属的账户
#SBATCH -J group-algo # 所运行的任务名称 (自己取)
#SBATCH -N 1 # 占用的节点数（根据代码要求确定）
#SBATCH --ntasks-per-node=3 # 运行的进程数（根据代码要求确定）
#SBATCH --cpus-per-task=1 # 每个进程的 CPU 核数 （根据代码要求确定）
#SBATCH --gres=gpu:2080ti:1 # 占用的 GPU 卡数 （根据代码要求确定）
#SBATCH -p prod # 任务运行所在的分区 (根据代码要求确定，gpu 为 gpu 分区，gpu4
#SBATCH -t 1-00:00:00 # 运行的最长时间 day-hour:minute:second
#SBATCH -o group_algo.out # 任务输出文件名字
#SBATCH -e group_algo.err #输出错误的文件
#SBATCH --mem=15G #任务所需要的内存数量，一定要指明内存大小！否则任务会独占整个机器的所有内存，导致任务排队

srun -l \
--container-image "/data1/liuyj/code/lyj_pytorch_12_14_nlp181.sqsh" \  
# --container-mounts "/data1/jqliu/DATA/:/home/data/:ro,/data1/liuyj/code/:/home/lyj/" \
# -container-mounts "/data1/jqliu/DATA/:/exp-data/:ro,/data1/liuyj/code/:/lyj-code/" \
--container-remap-root \
--container-writable \
# /bin/bash -c "cd /lyj-code/Group-algorithm/training && source activate && conda activate pytorch14 && python main.py --dataset cifar100 --model VGG16 --data_root /exp-data/cifar100 \
#                 --save_dir vgg16_cifar100_0.4_lr0.02_60120 --milestones 60 120 \
#                 --load_ck pretrain/guanfang_cifar100_vgg16.pth --batch_size 64 --max_epoch 200 --learning_rate 0.02 --final_threshold 0.4 \
#                 --mask &> vgg16_cifar100_0.4_lr0.02_60120.txt "
/bin/bash -c "ls"
set +x
