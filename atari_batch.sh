#!/bin/bash
#SBATCH --job-name=mujoco
#SBATCH --partition=gpu
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=40000M
#SBATCH --gpus-per-node=2
#SBATCH --nodes=3              
#SBATCH --ntasks=6           
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --account=ailab    

srun --ntasks=1 --nodes=1 --exclusive python train.py --tag='mujoco' --alg=rrd_mujoco_pytorch --basis_alg=sac --code=pytorch --rrd_bias_correction=True --env=Ant-v3 --rrd_batch_size=128 --rrd_sample_size=32  --train_batches=100  --chi2_coeff=0.0 --q_lr=3e-4 --polyak=0.999&

srun --ntasks=1 --nodes=1 --exclusive python train.py --tag='mujoco_2' --alg=rrd_mujoco_pytorch --basis_alg=sac --code=pytorch --rrd_bias_correction=True --env=Ant-v3 --rrd_batch_size=256 --rrd_sample_size=32  --train_batches=100  --chi2_coeff=0.1 --q_lr=3e-4 --polyak=0.999&

srun --ntasks=1 --nodes=1 --exclusive python train.py --tag='mujoco_3' --alg=rrd_mujoco_pytorch --basis_alg=sac --code=pytorch --rrd_bias_correction=True --env=Ant-v3 --rrd_batch_size=256 --rrd_sample_size=64  --train_batches=100  --chi2_coeff=0.2 --q_lr=3e-4 --polyak=0.999&

srun --ntasks=1 --nodes=1 --exclusive python train.py --tag='mujoco_4' --alg=rrd_mujoco_pytorch --basis_alg=sac --code=pytorch --rrd_bias_correction=True --env=Ant-v3 --rrd_batch_size=128 --rrd_sample_size=32  --train_batches=100  --chi2_coeff=0.3 --q_lr=3e-4 --polyak=0.999&

srun --ntasks=1 --nodes=1 --exclusive python train.py --tag='mujoco_5' --alg=rrd_mujoco_pytorch --basis_alg=sac --code=pytorch --rrd_bias_correction=True --env=Ant-v3 --rrd_batch_size=512 --rrd_sample_size=64  --train_batches=100  --chi2_coeff=0.4 --q_lr=3e-4 --polyak=0.999&

srun --ntasks=1 --nodes=1 --exclusive python train.py --tag='mujoco_6' --alg=rrd_mujoco_pytorch --basis_alg=sac --code=pytorch --rrd_bias_correction=True --env=Ant-v3 --rrd_batch_size=256 --rrd_sample_size=64  --train_batches=100  --chi2_coeff=0.5 --q_lr=3e-4 --polyak=0.999&

wait

