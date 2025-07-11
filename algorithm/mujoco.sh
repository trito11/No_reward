#!/bin/bash
#SBATCH --job-name=mujoco
#SBATCH --partition=gpu
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=10000M
#SBATCH --gpus-per-node=1
#SBATCH --nodes=3        
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=ailab    
source /home/tnguye11/anaconda3/bin/activate mujoco
module load cuda/11.8


srun --ntasks=1 --nodes=1 --exclusive python iq.py --seed=2 --re_coefficient=3 --loss_type=swap --r2_coefficient=0.005 --qf_a_coefficient=0.01&
srun --ntasks=1 --nodes=1 --exclusive python iq.py --seed=2 --re_coefficient=3 --loss_type=swap --r2_coefficient=0.01 --qf_a_coefficient=0.01&
srun --ntasks=1 --nodes=1 --exclusive python iq.py --seed=2 --re_coefficient=3 --loss_type=swap --r2_coefficient=0.01 --qf_a_coefficient=0.01&



wait
