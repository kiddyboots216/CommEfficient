#!/bin/bash

#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=4 # number of cores per task
# I think gpu:4 will request 4 of any kind of gpu per node,
# and gpu:v100_32:8 should request 8 v100_32 per node
#SBATCH --gres=gpu:8
#SBATCH --nodelist=steropes # if you need specific nodes
#SBATCH -t 7-2:00 # time requested (D-HH:MM)
# slurm will cd to this directory before running the script
# you can also just run sbatch submit.sh from the directory
# you want to be in
#SBATCH -D /data/ashwineep/CommEfficient/CommEfficient
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

# activate your virtualenv, do your conda magic, etc.
source activate comm-efficient

# do ALL the research
python fed_train.py --momentum_type virtual --error_type virtual --mode sketch --num_clients 5000 --participation 0.01 --num_devices 8 --k 50000 --num_rows 1 --num_cols 500000 --supervised --static_datasets

# print completion time
date
