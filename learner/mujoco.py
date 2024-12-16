import copy
import numpy as np
from envs import make_env
import time
import wandb
import math

class MuJoCoLearner:
    def __init__(self, args):
        self.ep_counter = 0
        self.step_counter = 0
        self.target_count = 0
        self.num_envs=args.num_envs
        self.learner_info = [
        ]
        self.ep = {}
        for i in range(self.num_envs):
            self.ep[i]=[]
    def learn(self, args, env, agent, buffer):

        env.last_obs = env.reset()
        for _ in range(args.iterations):
            obs = env.get_obs()
            for timestep in range(args.timesteps):
                obs_pre = obs
                action = agent.step(obs, explore=True)
                obs, reward, done, info = env.step(action)
                self.step_counter += self.num_envs
                for i in range(self.num_envs):
                    transition = {
                        'obs': obs_pre[i],
                        'obs_next': obs[i],
                        'acts': action[i],
                        'rews': reward[i] ,
                        'done': done[i] if env.step_buffer[i]<args.test_timesteps else False,
                        'real_done': done[i]
                    }
                    self.ep[i].append(transition)
                    if done[i]:
                        wandb.log({'Episode_reward':reward[i] })
                        env.sum_rews+=reward[i]
                        self.ep_counter += 1
                        wandb.log({'Train_step':len(self.ep[i]) })
                        for transition_ep in self.ep[i]:
                            transition = copy.deepcopy(transition_ep)
                            buffer.store_transition(transition)
                        self.ep[i] = []


            if buffer.step_counter>=args.warmup:
                if args.obs_normalization:
                    agent.normalizer_update(buffer.sample_batch())
                for _ in range(args.train_batches):
                    self.target_count += 1
                    if 'pi_delay_freq' in args.__dict__.keys():
                        batch = buffer.sample_batch()
                        info = agent.train(batch,train_q=True)
                        args.logger.add_dict(info)
                        if self.target_count%args.pi_delay_freq==0:
                            batch = buffer.sample_batch()
                            info = agent.train(batch,train_policy=True)
                            args.logger.add_dict(info)
                    else:
                        batch = buffer.sample_batch()
                        info = agent.train(batch,train_q=True,train_policy=True)
                        args.logger.add_dict(info)
                    if self.target_count%args.train_target==0:
                        agent.update_target_net()

        args.logger.add_dict(env.env_info)

