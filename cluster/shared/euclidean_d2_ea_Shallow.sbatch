#!/bin/bash
#SBATCH --job-name=eeg_ea_shared_d2_m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-user=a.bruno@ufabc.edu.br 
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --output=/mnt/beegfs/home/aristimunha/bruna/EEG_EA-Domain-Adap/logs/eeg_ea_shared_d2_m3_%j.log

module load /mnt/beegfs/home/aristimunha/bruna/EEG_EA-Domain-Adap/cluster/python

python /mnt/beegfs/home/aristimunha/bruna/EEG_EA-Domain-Adap/src/Bruna/script_moabb_exp1.py --config_file=/mnt/beegfs/home/aristimunha/bruna/EEG_EA-Domain-Adap/config/config_Shallow.yaml  --num_exp 'exp_1' --dataset 'Schirrmeister2017' --ea 'alignment'


 

