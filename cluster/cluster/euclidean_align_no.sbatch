#!/bin/bash
#SBATCH --job-name=eeg_euclidean_with_alignment_exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --mail-user=a.bruno@ufabc.edu.br
#SBATCH --mail-type=ALL		


module load /mnt/beegfs/home/chevallier/bruno/Ensemble-Transfer-Learning-EEG/cluster/python

python /mnt/beegfs/home/chevallier/bruno/Ensemble-Transfer-Learning-EEG/src/run_baseline.py --config_file=/mnt/beegfs/home/chevallier/bruno/Ensemble-Transfer-Learning-EEG/config/config.yaml --data=alignment

 

