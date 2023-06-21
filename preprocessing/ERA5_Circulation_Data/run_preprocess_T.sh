#!/bin/bash
#PBS -N T
#PBS -l nodes=1:ppn=1
#PBS -l walltime=480:00:00
#PBS -V

source activate NCL

cd /home/shixm/Workspace/SDD_Part1/ERA5_preprocess
which ncl >& preprocess_T.log
ncl preprocessERA5_temperature.ncl >& preprocess_T.log



