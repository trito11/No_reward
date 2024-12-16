#!/bin/bash
#SBATCH --job-name=mujoco
#SBATCH --partition=gpulong 
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=240:00:00
#SBATCH --mem=10000M
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1             
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --account=ailab    
source /home/tnguye11/anaconda3/bin/activate RRD
module load cuda/11.8

srun --ntasks=1 --nodes=1 --exclusive python train.py --tag='mujoco_xql_q_q' --alg=rrd_mujoco_pytorch_xql --basis_alg=sac --code=pytorch --rrd_bias_correction=True --env=Ant-v2 --num_envs=1 --rrd_batch_size=256 --rrd_sample_size=64  --train_batches=100  --chi2_coeff=0.2 --chi2_coeff1=0.5 --q_lr=3e-4 --polyak=0.995 --train_target=1



