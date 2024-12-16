#!/bin/bash
#SBATCH --job-name=Atari_para
#SBATCH --partition=gpu
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=20:00:00
#SBATCH --mem=20000M
#SBATCH --gpus-per-node=4
#SBATCH --nodes=2               # Number of nodes
#SBATCH --ntasks=2              # Number of parallel tasks
#SBATCH --cpus-per-task=4 


# Run training for the first configuration
srun --exclusive -N 1 -n 1  python train.py --tag='Atari_1024_32' --alg=rrd_atari_pytorch --basis_alg=dqn --code=pytorch --dueling=True --rrd_bias_correction=True --env=Assault --rrd_batch_size=1024 --rrd_sample_size=32 --alpha=0.95 --train_batches=1 

# Run training for the second configuration
srun --exclusive -N 1 -n 1  python train.py --tag='Atari_32_32' --alg=rrd_atari --basis_alg=dqn --rrd_bias_correction=False --env=Assault --rrd_batch_size=32 --rrd_sample_size=32&

# Wait for both background jobs to finish
wait
