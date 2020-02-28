#!/bin/bash

#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=48 # number of cores per task
# I think gpu:4 will request 4 of any kind of gpu per node,
# and gpu:v100_32:8 should request 8 v100_32 per node
#SBATCH --gres=gpu:1
##SBATCH --nodelist=pavia # if you need specific nodes
#SBATCH --exclude=atlas,blaze,steropes # nodes not yet on SLURM-only
#SBATCH -t 2-2:00 # time requested (D-HH:MM)
# slurm will cd to this directory before running the script
# you can also just run sbatch submit.sh from the directory
# you want to be in
##SBATCH -D /home/eecs/drothchild/slurm
# use these two lines to control the output file. Default is
# slurm-<jobid>.out. By default stdout and stderr go to the same
# place, but if you use both commands below they'll be split up
# filename patterns here: https://slurm.schedmd.com/sbatch.html
# %N is the hostname (if used, will create output(s) per node)
# %j is jobid
##SBATCH -o slurm.%N.%j.out # STDOUT
##SBATCH -e slurm.%N.%j.err # STDERR
# if you want to get emails as your jobs run/fail
##SBATCH --mail-type=NONE # Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH --mail-user=<your_email> # Where to send mail 

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
conda activate comm
ulimit -n 50000

# python will buffer output of your script unless you set this
# if you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed
export PYTHONUNBUFFERED=1

# do ALL the research
rsync -zarh --exclude ".git/*" --exclude "*.out" ~/CommEfficient /data/ashwineep/
cd /data/ashwineep/CommEfficient/CommEfficient
OMP_NUM_THREADS=16 KMP_INIT_AT_FORK=FALSE python cv_train.py \
    --dataset_dir /data/ashwineep/datasets/${1}/ \
    --valid_batch_size 512 \
    --tensorboard \
    --dataset_name ${1} \
    --model ${2} \
    --mode ${3} \
    --num_clients ${4} \
    --num_workers ${5} \
    --local_batch_size ${6} \
    --error_type ${7} \
    --num_epochs ${8} \
    --pivot_epoch ${9} \
    --lr_scale ${10} \
    --local_momentum ${11} \
    --virtual_momentum ${12} \
    --weight_decay 5e-4 \
    --num_fedavg_epochs ${13} \
    --fedavg_lr_decay 0 \
    --fedavg_batch_size ${14} \
    --num_devices 1 \
    --k ${15} \
    --num_rows 1 \
    --num_cols ${16} \
    --share_ps_gpu \
    --port ${17} \
    --train_dataloader_workers 2 \
    --val_dataloader_workers 0 \
    --seed ${18} \
    --mal_targets ${19} \
    --mal_boost ${20} \
    --mal_num_clients ${21} \
    --mal_epoch ${22} \
    --mal_type ${23} \
    --noise_multiplier ${24} \
    --l2_norm_clip ${25} \
    ${26} \
    ${27} \
    ${28} \
    ${29} \

# print completion time
date
