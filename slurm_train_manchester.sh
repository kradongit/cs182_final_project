#!/usr/bin/env bash

#SBATCH -D /home/eecs/krad/final_project_cs182/
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --nodelist=bombe
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:1
#SBATCH -o /home/eecs/krad/final_project_cs182/slurm_log/bombe/slurm.%N.%j.out # STDOUT
#SBATCH -e /home/eecs/krad/final_project_cs182/slurm_log/bombe/slurm.%N.%j.err # STDOUT

source ~/.bashrc
cd ~/final_project_cs182/ || exit
conda activate cs182

python train_fruitbot.py --config configurations/ppo_impala_100_cuda.yaml
