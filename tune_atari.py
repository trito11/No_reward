import os
import subprocess
import time
import argparse
import random
import glob
import numpy as np
import concurrent.futures
parser = argparse.ArgumentParser()


parser.add_argument('--task', type=str, default='run_exps', choices=['run_exps','collect_results'])
parser.add_argument('--env', type=str, default='Assault', choices=['Assault'])
parser.add_argument('--dueling', type=str, default='True')
parser.add_argument('--rrd_bias_correction', type=str, default='True')
parser.add_argument('--basis_alg', type=str, default='dqn', choices=['dqn','sac'])
parser.add_argument('--models_per_gpu', type=int, default=4)
parser.add_argument('--gpus', nargs='+', type=int, default=[0,1,2], help='gpus indices used for multi_gpu')

parser.add_argument('--result_dir', type=str, default='results/stock_d=21_a=8_pi=eps-greedy0.1_std=0.1', help='result directory for collect_results()')
args = parser.parse_args()

def multi_gpu_launcher(commands,gpus,models_per_gpu):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    procs = [None]*len(gpus)*models_per_gpu

    while len(commands) > 0:
        for i,proc in enumerate(procs):
            gpu_idx = gpus[i % len(gpus)]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this index; launch a command.
                cmd = commands.pop(0)
                print(cmd)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs[i] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs:
        if p is not None:
            p.wait()

def create_commands(env='Assault',dueling='True',rrd_bias_correction='True',basis_alg='dqn'):
    hyper_mode = 'full' # ['full', 'best']
    if hyper_mode == 'full':
        # Grid search space: used for grid search in the paper
        q_lr = [1e-4,0.625e-4]
        train_mode_space = [(128,32)]
        chi2_coeff = [0,0.1,0.,0.5]
        polyak = [0.995, 0.999]
        train_batch=[20]
        alpha=[0.1,0.2,1]



    elif hyper_mode == 'best':
        # The best searc sapce inferred fro prior experiments 
        lr_space = [1e-4]
        train_mode_space = [(1,1,1)]
        beta_space = [1]
        rbfsigma_space = [10] #10 for mnist, 0.1 for mushroom

    commands = []
    if basis_alg=='dqn':
        for lr in q_lr:
            for chi2 in chi2_coeff:
                for batch_size,num_sample in train_mode_space:
                    for polyak_lr in polyak:
                        for train_batch_lr in train_batch:
                            commands.append(f'python train.py --tag={env} --alg=rrd_atari_pytorch --basis_alg={basis_alg} --code=pytorch --dueling={dueling} --rrd_bias_correction={rrd_bias_correction} --env={env} --rrd_batch_size={batch_size} --rrd_sample_size={num_sample}  --train_batches={train_batch_lr}  --chi2_coeff={chi2} --q_lr={lr} --polyak={polyak_lr}')
    elif basis_alg=='sac':
            for lr in q_lr:
                for chi2 in chi2_coeff:
                    for batch_size,num_sample in train_mode_space:
                        for polyak_lr in polyak:
                            for train_batch_lr in train_batch:
                                for alpha_lr in alpha:
                                    commands.append(f'python train.py --tag={env} --alg=rrd_atari_pytorch --basis_alg={basis_alg} --code=pytorch --dueling={dueling} --rrd_bias_correction={rrd_bias_correction} --env={env} --rrd_batch_size={batch_size} --rrd_sample_size={num_sample}  --train_batches={train_batch_lr}  --chi2_coeff={chi2} --q_lr={lr} --polyak={polyak_lr} --alpha={alpha_lr}')


    else:
        raise NotImplementedError(f"Running experiments with: env={env}, dueling={dueling}, rrd_bias_correction={rrd_bias_correction}, basis_alg={basis_alg}")
    return commands


def run_exps():
    commands = []
    commands += create_commands(args.env, args.dueling, args.rrd_bias_correction, args.basis_alg)
    random.shuffle(commands)
    multi_gpu_launcher(commands, args.gpus, args.models_per_gpu)

def collect_results():
    filenames = glob.glob(os.path.join(args.result_dir,"*.npz"))
    results = {}
    for filename in filenames:
        k = np.load(filename)
        regret = k['arr_0'][:,1,:]
        regret = np.min(regret,1) # best regret of a run
        regret = np.mean(regret)
        results[filename] = regret
    
    filenames.sort(key=lambda x: results[x])
    
    for filename in filenames:
        print('{}:   {}'.format(filename,results[filename]))

if __name__ == '__main__':
    eval(args.task)()

