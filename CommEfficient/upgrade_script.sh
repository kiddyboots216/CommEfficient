#!/bin/bash

#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=48 # number of cores per task
# I think gpu:4 will request 4 of any kind of gpu per node,
# and gpu:v100_32:8 should request 8 v100_32 per node
#SBATCH -t 2-2:00 # time requested (D-HH:MM)
# slurm will cd to this directory before running the script
# you can also just run sbatch submit.sh from the directory
# you want to be in
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

# python will buffer output of your script unless you set this
# if you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed
export PYTHONUNBUFFERED=1

# do ALL the research
#rsync -zarh --exclude ".git/*" --exclude "*.out" ~/CommEfficient /data/ashwineep/
conda update -y -c pytorch pytorch torchvision
# print completion time
date

