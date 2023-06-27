# python main.py --dataset cifar10 --model VGG16 --data_root /home/data/cifar10 --save_dir vgg16_0.75 --milestones 60 80 --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 4 --final_threshold 0.75 --mask &> vgg16_0.75.txt &

# python main.py --dataset tiny-imagenet --model VGG16 --data_root /home/data/tiny_imagenet/tiny-imagenet-200 --save_dir tim_vgg16_0.75 --milestones 60 120 --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 4 --final_threshold 0.75 --mask &> vgg16_tim_0.75.txt &

# python main.py --dataset cifar100 --model VGG16 --data_root /home/data/cifar100 --save_dir vgg16_cifar100_0.9_lr0.1_60120 --milestones 60 120 \
#                 --batch_size 64 --max_epoch 200 --learning_rate 0.1 --gpu 5 --final_threshold 0.9 --mask &> vgg16_cifar100_0.9_lr0.1_60120.txt

# python main.py --dataset cifar100 --model VGG16 --data_root /home/data/cifar100 --save_dir vgg16_cifar100_0.8_lr0.1_60120 --milestones 60 120 \
#                  --batch_size 64 --max_epoch 200 --learning_rate 0.1 --gpu 5 --final_threshold 0.8 --mask &> vgg16_cifar100_0.8_lr0.1_60120.txt

# python main.py --dataset cifar100 --model VGG16 --data_root /home/data/cifar100 --save_dir vgg16_cifar100_0.7_lr0.02_60120 --milestones 60 120 \
#                --load_ck pretrain/guanfang_cifar100_vgg16.pth --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.7\
# 	       --mask &> vgg16_cifar100_0.7_lr0.02_60120.txt

# python main.py --dataset cifar100 --model VGG16 --data_root /home/data/cifar100 --save_dir vgg16_cifar100_0.6_lr0.02_60120 --milestones 60 120 \
#                --load_ck pretrain/guanfang_cifar100_vgg16.pth --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.6 \
# 	       --mask &> vgg16_cifar100_0.6_lr0.02_60120.txt

# python main.py --dataset cifar100 --model VGG16 --data_root /home/data/cifar100 --save_dir vgg16_cifar100_0.5_lr0.02_60120 --milestones 60 120 \
#                 --load_ck pretrain/guanfang_cifar100_vgg16.pth --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.5 \
# 	       	--mask &> vgg16_cifar100_0.5_lr0.02_60120.txt

# python main.py --dataset cifar100 --model VGG16 --data_root /home/data/cifar100 --save_dir vgg16_cifar100_0.4_lr0.02_60120 --milestones 60 120 \
#                 --load_ck pretrain/guanfang_cifar100_vgg16.pth --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.4 \
#                 --mask &> vgg16_cifar100_0.4_lr0.02_60120.txt

# python main.py --dataset cifar100 --model VGG16 --data_root /home/data/cifar100 --save_dir vgg16_cifar100_0.3_lr0.02_60120 --milestones 60 120 \
#                 --load_ck pretrain/guanfang_cifar100_vgg16.pth --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.3 \
#                 --mask &> vgg16_cifar100_0.3_lr0.02_60120.txt

# python main.py --dataset cifar100 --model VGG16 --data_root /home/data/cifar100 --save_dir vgg16_cifar100_0.2_lr0.02_60120 --milestones 60 120 \
#                  --load_ck pretrain/guanfang_cifar100_vgg16.pth --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.2 \
#                  --mask &> vgg16_cifar100_0.2_lr0.02_60120.txt

# python main.py --dataset cifar100 --model VGG16 --data_root /home/data/cifar100 --save_dir vgg16_cifar100_0.1_lr0.02_60120 --milestones 60 120 \
#                  --load_ck pretrain/guanfang_cifar100_vgg16.pth --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.1 \
#                  --mask &> vgg16_cifar100_0.1_lr0.02_60120.txt

# python main.py --dataset cifar100 --model VGG16 --data_root /home/data/cifar100 --save_dir vgg16_cifar100_0.08_lr0.02_60120 --milestones 60 120 \
#                  --load_ck pretrain/guanfang_cifar100_vgg16.pth --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.08 \
#                  --mask &> vgg16_cifar100_0.08_lr0.02_60120.txt

# python main.py --dataset cifar100 --model VGG16 --data_root /home/data/cifar100 --save_dir vgg16_cifar100_0.06_lr0.02_60120 --milestones 60 120 \
#                  --load_ck pretrain/guanfang_cifar100_vgg16.pth --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.06 \
#                  --mask &> vgg16_cifar100_0.06_lr0.02_60120.txt

# python main.py --dataset cifar100 --model VGG16 --data_root /home/data/cifar100 --save_dir vgg16_cifar100_0.04_lr0.02_60120 --milestones 60 120 \
#                  --load_ck pretrain/guanfang_cifar100_vgg16.pth --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.04 \
#                  --mask &> vgg16_cifar100_0.04_lr0.02_60120.txt


# python main.py --dataset cifar100 --model VGG16 --data_root /home/data/cifar100 --save_dir vgg16_cifar100_0.4_lr0.02_60120 --milestones 60 120 --load_ck pretrain/guanfang_cifar100_vgg16.pth --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.4 --mask &> vgg16_cifar100_0.4_lr0.02_60120.txt


python main.py --dataset cifar10 --model VGG16-16 --data_root /home/data/cifar100 --save_dir vgg16-16_cifar10_0.1_lr0.02_60 --milestones 60 \
                 --batch_size 64 --max_epoch 100 --learning_rate 0.02 --gpu 2 --final_threshold 0.1 \
                 --mask &> vgg16-16_cifar10_0.1_lr0.02_60.txt

python main.py --dataset cifar100 --model VGG16-16 --data_root /home/data/cifar100 --save_dir vgg16-16_cifar100_nop_lr0.1_60 --milestones 60 \
                 --batch_size 64 --max_epoch 100 --learning_rate 0.1 --gpu 2 &> vgg16-16_cifar100_nop_lr0.1_60.txt