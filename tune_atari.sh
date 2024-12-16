#!/bin/bash
#SBATCH --job-name=Atari_para
#SBATCH --partition=gpu
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=60000M
#SBATCH --gpus-per-node=3
#SBATCH --nodes=2             
#SBATCH --ntasks=2           
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --account=ailab    

source /home/tnguye11/anaconda3/bin/activate RRD
module load cuda/11.8

commands=(
    "python tune_atari.py --dueling=True --rrd_bias_correction=False  --basis_alg=dqn --gpus 0 1 2 "
    "python tune_atari.py --dueling=True --rrd_bias_correction=False  --basis_alg=sac --gpus 0 1 2 "
)


for i in "${!commands[@]}"; do
   srun --exclusive -N1 -n1 ${commands[i]} &
done

wait
