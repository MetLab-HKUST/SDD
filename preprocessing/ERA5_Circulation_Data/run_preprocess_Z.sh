#!/bin/bash
#PBS -N Z
#PBS -l nodes=1:ppn=1
#PBS -l walltime=480:00:00
#PBS -V

source activate NCL

cd /home/shixm/Workspace/SDD_Part1/ERA5_preprocess
ncl preprocessERA5_geopotential.ncl >& preprocess_Z.log



