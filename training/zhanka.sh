#!/bin/bash
#SBATCH -A lyj06
#SBATCH -J occ_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=prod
#SBATCH --gres=gpu:1
#SBATCH --time=10-0:00:00
#SBATCH --mem=5m
#SBATCH -q normal

scontrol show job -d $SLURM_JOB_ID
sleep 10d
