#!/bin/bash

#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=23:59:00

python digit_pytorch/train_teacher.py