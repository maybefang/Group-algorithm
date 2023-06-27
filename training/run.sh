

# python main.py --dataset tiny-imagenet --model VGG19 --data_root /home/data/tiny_imagenet/tiny-imagenet-200 --save_dir vgg19_0.8_min0_lr0.02_60120 --milestones 60 120 --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 4 --final_threshold 0.8 --mask &> vgg19_tim_0.8_min0_lr02_60120.txt

#python main.py --dataset tiny-imagenet --model VGG19 --data_root /home/data/tiny_imagenet/tiny-imagenet-200 --save_dir vgg19_0.7_lr0.02 --milestones 60 --batch_size 64 --max_epoch 100 --learning_rate 0.02 --gpu 5 --final_threshold 0.7 &> vgg19_tim_0.7.txt

#python main.py --dataset tiny-imagenet --model VGG19 --data_root /home/data/tiny_imagenet/tiny-imagenet-200 --save_dir vgg19_0.6_lr0.02 --milestones 60 --batch_size 64 --max_epoch 100 --learning_rate 0.02 --gpu 5 --final_threshold 0.6 &> vgg19_tim_0.6.txt

#python main.py --dataset cifar100 --model ResNet50 --data_root /home/data/tiny_imagenet/tiny-imagenet-200 --save_dir resnet50_tiny_nop_lr0.2_60120 --milestones 60 120 --batch_size 64 --max_epoch 200 --learning_rate 0.2 --gpu 5 --final_threshold 1 &> resnet50_tiny_nop_60120.txt 

#python main.py --dataset cifar100 --model ResNet18 --data_root /home/data/cifar100 --save_dir resnet18_cifar100_0.9_lr0.02_60120 --milestones 60 120 --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.9 --mask &> resnet18_cifar100_0.9_60120.txt

#python main.py --dataset cifar100 --model ResNet18 --data_root /home/data/cifar100 --save_dir resnet18_cifar100_0.8_lr0.02_60120 --milestones 60 120 --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.8 --mask &> resnet18_cifar100_0.8_60120.txt

#python main.py --dataset cifar100 --model ResNet18 --data_root /home/data/cifar100 --save_dir resnet18_cifar100_0.7_lr0.02_60120 --milestones 60 120 --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.7 --mask &> resnet18_cifar100_0.7_60120.txt

#python main.py --dataset cifar100 --model ResNet18 --data_root /home/data/cifar100 --save_dir resnet18_cifar100_0.6_lr0.02_60120 --milestones 60 120 --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.6 --mask &> resnet18_cifar100_0.6_60120.txt

#python main.py --dataset cifar100 --model ResNet18 --data_root /home/data/cifar100 --save_dir resnet18_cifar100_0.5_lr0.02_60120 --milestones 60 120 --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.5 --mask &> resnet18_cifar100_0.5_60120.txt 

#python main.py --dataset cifar100 --model ResNet18 --data_root /home/data/cifar100 --save_dir resnet18_cifar100_0.4_lr0.02_60120 --milestones 60 120 --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.4 --mask &> resnet18_cifar100_0.4_60120.txt

#python main.py --dataset cifar100 --model ResNet18 --data_root /home/data/cifar100 --save_dir resnet18_cifar100_0.3_lr0.02_60120 --milestones 60 120 --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.3 --mask &> resnet18_cifar100_0.3_60120.txt

#python main.py --dataset cifar100 --model ResNet18 --data_root /home/data/cifar100 --save_dir resnet18_cifar100_0.2_lr0.02_60120 --milestones 60 120 --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.2 --mask &> resnet18_cifar100_0.2_60120.txt

#python main.py --dataset cifar100 --model ResNet18 --data_root /home/data/cifar100 --save_dir resnet18_cifar100_0.1_lr0.02_60120 --milestones 60 120 --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.1 --mask &> resnet18_cifar100_0.1_60120.txt

#python main.py --dataset cifar100 --model ResNet18 --data_root /home/data/cifar100 --save_dir resnet18_cifar100_0.08_lr0.02_60120 --milestones 60 120 --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.08 --mask &> resnet18_cifar100_0.0_60120.txt

#python main.py --dataset cifar100 --model ResNet18 --data_root /home/data/cifar100 --save_dir resnet18_cifar100_0.06_lr0.02_60120 --milestones 60 120 --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.06 --mask &> resnet18_cifar100_0.06_60120.txt

#python main.py --dataset cifar100 --model ResNet18 --data_root /home/data/cifar100 --save_dir resnet18_cifar100_0.04_lr0.02_60120 --milestones 60 120 --batch_size 64 --max_epoch 200 --learning_rate 0.02 --gpu 5 --final_threshold 0.04 --mask &> resnet18_cifar100_0.04_60120.txt



# python main.py --dataset tiny-imagenet --model VGG19 --data_root /home/data/tiny_imagenet/tiny-imagenet-200 --save_dir vgg19_0.4_lr0.02 --milestones 60 \
#                --load_ck pretrain/guanfang_trans_vgg19.pth --batch_size 64 --max_epoch 100 --learning_rate 0.02 --lamda_min 1 --final_lambda 10 --gpu 5 \
#                --final_threshold 0.4 --mask &> vgg19_tim_0.4_lr02_60.txt 

# python main.py --dataset tiny-imagenet --model VGG19 --data_root /home/data/tiny_imagenet/tiny-imagenet-200 --save_dir vgg19_0.3_lr0.02 --milestones 60 \
#                --load_ck pretrain/guanfang_trans_vgg19.pth --batch_size 64 --max_epoch 100 --learning_rate 0.02 --lamda_min 1 --final_lambda 10 --gpu 5 \
#                --final_threshold 0.3 --mask &> vgg19_tim_0.3_lr02_60.txt 

# python main.py --dataset tiny-imagenet --model VGG19 --data_root /home/data/tiny_imagenet/tiny-imagenet-200 --save_dir vgg19_0.2_lr0.02 --milestones 60 \
#                --load_ck pretrain/guanfang_trans_vgg19.pth --batch_size 64 --max_epoch 100 --learning_rate 0.02 --lamda_min 1 --final_lambda 10 --gpu 5 \
#                --final_threshold 0.2 --mask &> vgg19_tim_0.2_lr02_60.txt 

# python main.py --dataset tiny-imagenet --model VGG19 --data_root /home/data/tiny_imagenet/tiny-imagenet-200 --save_dir vgg19_0.6_lr0.02 --milestones 60 \
#                --load_ck pretrain/guanfang_trans_vgg19.pth --batch_size 64 --max_epoch 100 --learning_rate 0.02 --lamda_min 1 --final_lambda 10 --gpu 5 \
#                --final_threshold 0.6 --mask &> vgg19_tim_0.6_lr02_60.txt 

# python main.py --dataset tiny-imagenet --model VGG19 --data_root /home/data/tiny_imagenet/tiny-imagenet-200 --save_dir vgg19_0.5_lr0.02 --milestones 60 \
#                --load_ck pretrain/guanfang_trans_vgg19.pth --batch_size 64 --max_epoch 100 --learning_rate 0.02 --lamda_min 1 --final_lambda 10 --gpu 5 \
#                --final_threshold 0.5 --mask &> vgg19_tim_0.5_lr02_60.txt 


python main.py --dataset tiny-imagenet --model VGG19 --data_root /home/data/tiny_imagenet/tiny-imagenet-200 --save_dir vgg19_0.1_lr0.02 --milestones 60 \
               --load_ck pretrain/guanfang_trans_vgg19.pth --batch_size 64 --max_epoch 100 --learning_rate 0.02 --lamda_min 50 --final_lambda 60 --gpu 5 \
               --final_threshold 0.1 --mask &> vgg19_tim_0.1_lr02_60.txt 

python main.py --dataset tiny-imagenet --model ResNet50 --data_root /home/data/tiny_imagenet/tiny-imagenet-200 --save_dir resnet50_0.8_lr0.02 --milestones 60 \
               --load_ck pretrain/guanfang_trans_resnet50.pth --batch_size 64 --max_epoch 100 --learning_rate 0.02 --lamda_min 50 --final_lambda 60 --gpu 5 \
               --final_threshold 0.8 --mask &> resnet50_tim_0.8_lr02_60.txt 

python main.py --dataset tiny-imagenet --model ResNet50 --data_root /home/data/tiny_imagenet/tiny-imagenet-200 --save_dir resnet50_0.7_lr0.02 --milestones 60 \
               --load_ck pretrain/guanfang_trans_resnet50.pth --batch_size 64 --max_epoch 100 --learning_rate 0.02 --lamda_min 50 --final_lambda 60 --gpu 5 \
               --final_threshold 0.7 --mask &> resnet50_tim_0.7_lr02_60.txt 

python main.py --dataset tiny-imagenet --model ResNet50 --data_root /home/data/tiny_imagenet/tiny-imagenet-200 --save_dir resnet50_0.6_lr0.02 --milestones 60 \
               --load_ck pretrain/guanfang_trans_resnet50.pth --batch_size 64 --max_epoch 100 --learning_rate 0.02 --lamda_min 50 --final_lambda 60 --gpu 5 \
               --final_threshold 0.6 --mask &> resnet50_tim_0.6_lr02_60.txt 









