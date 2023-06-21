#!/bin/bash
#PBS -N MaxTF
#PBS -l nodes=1:ppn=1
#PBS -l walltime=480:00:00
#PBS -V

export PATH=/home/shixm/anaconda3/bin:$PATH

source activate TF28

cd /home/shixm/Workspace/SDD_Part1/DL_Training/

export PYTHONUNBUFFERED=TRUE

python Train_RxNet_MaxOnly_SingleGPU.py >& train_max_only.log



