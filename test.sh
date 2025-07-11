#!/bin/bash
#SBATCH --job-name=Atari_para
#SBATCH --partition=gpu
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=3000M
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1              
#SBATCH --ntasks=1           
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=ailab    

source /home/tnguye11/anaconda3/bin/activate RRD
module load cuda/11.8
srun python a.py
