#!/bin/bash

#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=2G

python digit_pytorch/pate.py