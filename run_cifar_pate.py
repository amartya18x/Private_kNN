#!/bin/bash

#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --gpus=quadro_rtx_6000:1
#SBATCH --mem-per-cpu=2G

python digit_pytorch/pate.py