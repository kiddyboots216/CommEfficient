#!/bin/bash

#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=4 # number of cores per task
# I think gpu:4 will request 4 of any kind of gpu per node,
# and gpu:v100_32:8 should request 8 v100_32 per node
#SBATCH --gres=gpu:0
#SBATCH -t 2-2:00 # time requested (D-HH:MM)
# print some info for context
pwd
hostname
date

echo starting job...

# activate your virtualenv
# source /data/drothchild/virtualenvs/pytorch/bin/activate
# or do your conda magic, etc.
source ~/.bashrc
#conda init
conda activate afl
ulimit -n 50000

# python will buffer output of your script unless you set this
# if you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed
export PYTHONUNBUFFERED=1

# do ALL the research
rsync -zarh --exclude ".git/*" --exclude "*.out" ~/CommEfficient /data/scsi/ashwineep/
OMP_NUM_THREADS=16 cd /data/scsi/ashwineep/CommEfficient/CommEfficient

python certify.py --bound ${1}
