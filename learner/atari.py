import copy
import numpy as np
from envs import make_env
import time
import wandb
import math

class AtariLearner:
    def __init__(self, args):
        self.ep_counter = 0
        self.step_counter = 0
        self.target_count = 0
        self.learner_info = [
            'Epsilon'
        ]
        self.num_envs=args.num_envs
        self.ep = {}
        for i in range(self.num_envs):
            self.ep[i]=[]
        
        

        args.eps_act = args.eps_l
        # self.eps_decay = (args.eps_l-args.eps_r)/args.eps_decay

    def learn(self, args, env, agent, buffer):
        env.last_obs = env.reset()
        for _ in range(args.iterations):
            obs = env.get_obs()
            
            for timestep in range(args.timesteps):
                obs_pre = obs
                action = agent.step(obs, explore=True)
                args.eps_act = args.eps_r + (args.eps_l - args.eps_r) * math.exp(-1. * self.step_counter / args.eps_decay)
                wandb.log({'Epsilon':args.eps_act })
                obs, reward, done, _ = env.step(action)
                # print(obs)
                self.step_counter += self.num_envs
                for i in range(self.num_envs):
                    transition = {
                        'obs': obs_pre[i],
                        'obs_next': obs[i],
                        'acts': action[i],
                        'rews': reward[i] if args.env_type=='ep_rews' else np.clip(reward[i], -args.rews_scale, args.rews_scale),
                        'done': done[i]
                    }
                    self.ep[i].append(transition)
                    if done[i]:
                        wandb.log({'Train_step':len(self.ep[i]) })
                        wandb.log({'Episode_reward':reward[i] })
                        env.sum_rews+=reward[i]
                        self.ep_counter += 1
                        for transition_ep in self.ep[i]:
                            transition = copy.deepcopy(transition_ep)
                            buffer.store_transition(transition)
                        self.ep[i] = []

                    # print(f'Add bufer time: {a-time.time()}')
            args.logger.add_record('Epsilon', args.eps_act)


            if buffer.step_counter>=args.warmup:
                if args.obs_normalization:
                    agent.normalizer_update(buffer.sample_batch())
                # a=time.time()
                for _ in range(args.train_batches):
                    info = agent.train(buffer.sample_batch())
                    args.logger.add_dict(info)


        args.logger.add_dict(env.env_info)


