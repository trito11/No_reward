
## Requirements
Python 3.8.20
## Running Commands

Run the following commands to reproduce main results in [Learning Long-Term Reward Redistribution via Randomized Return Decomposition](https://arxiv.org/abs/2111.13485)
```bash
python train.py --tag='RRD Ant-v2' --alg=rrd --basis_alg=sac --env=Ant-v2
python train.py --tag='RRD-L(RD) Ant-v2' --alg=rrd --basis_alg=sac --rrd_bias_correction=True --env=Ant-v2
```

The following commands to switch the back-end algorithm of RRD.

```bash
python train.py --tag='RRD-TD3 Ant-v2' --alg=rrd --basis_alg=td3 --env=Ant-v2
python train.py --tag='RRD-DDPG Ant-v2' --alg=rrd --basis_alg=ddpg --env=Ant-v2
```

We include an *unofficial* implementation of IRCR for the ease of baseline comparison.  
Please refer to [tgangwani/GuidanceRewards](https://github.com/tgangwani/GuidanceRewards) for the official implementation of IRCR.

```bash
python train.py --tag='IRCR-SAC Ant-v2' --alg=ircr --basis_alg=sac --env=Ant-v2
python train.py --tag='IRCR-TD3 Ant-v2' --alg=ircr --basis_alg=td3 --env=Ant-v2
python train.py --tag='IRCR-DDPG Ant-v2' --alg=ircr --basis_alg=ddpg --env=Ant-v2
```

The following commands support the experiments on Atari games with episodic rewards.  

```bash
python train.py --tag='RRD-DQN Assault' --alg=dqn  --env=Assault
python train.py --tag='IRCR-DQN Assault' --alg=ircr --basis_alg=dqn --env=Assault
```

Use "envpool.make()" to create multi-env in ./envs


Run the following commands to use Xql algorithm
```bash
python train.py --tag='mujoco_xql_q_q' --alg=rrd_mujoco_pytorch_xql --basis_alg=sac --code=pytorch --rrd_bias_correction=True --env=Ant-v2 --num_envs=1 --rrd_batch_size=256 --rrd_sample_size=64  --train_batches=100  --chi2_coeff=0.2 --chi2_coeff1=0.5 --q_lr=3e-4 --polyak=0.995 --train_target=1
```

Change --alg=rrd_mujoco_pytorch to use Double-Q SAC algorithm

Change --alg=rrd_mujoco_pytorch_q to use Single-Q SAC algorithm

