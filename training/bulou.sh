
python main.py --dataset cifar100 --model VGG16 --data_root /home/data/cifar100 --save_dir vgg16_cifar100_0.2_lr0.02_60120 --milestones 60 120 \
                 --load_ck pretrain/guanfang_cifar100_vgg16.pth --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 2 --final_threshold 0.2 \
                 --mask &> vgg16_cifar100_0.2_lr0.02_60120_bulou.txt

python main.py --dataset cifar100 --model VGG16 --data_root /home/data/cifar100 --save_dir vgg16_cifar100_0.1_lr0.02_60120 --milestones 60 120 \
                 --load_ck pretrain/guanfang_cifar100_vgg16.pth --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 2 --final_threshold 0.1 \
                 --mask &> vgg16_cifar100_0.1_lr0.02_60120_bulou.txt












