#!/bin/bash
#SBATCH --job-name=ft_indiv_d1_m1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-user=a.bruno@ufabc.edu.br 
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --output=/mnt/beegfs/home/aristimunha/bruna/EEG_EA-Domain-Adap/logs/ft_indiv_d1_m1_%j.log

module load /mnt/beegfs/home/aristimunha/bruna/EEG_EA-Domain-Adap/cluster/python

python /mnt/beegfs/home/aristimunha/bruna/EEG_EA-Domain-Adap/src/Bruna/fine_tuning_exp4.py --config_file=/mnt/beegfs/home/aristimunha/bruna/EEG_EA-Domain-Adap/config/config.yaml  --num_exp 'exp_4' --dataset 'BNCI2014001' --ea 'no-alignment'


 

