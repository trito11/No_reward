import cv2
import gym
import numpy as np
from utils.os_utils_tf import remove_color

import envpool
class AtariNormalEnv():
    def __init__(self, args):
        self.args = args
        if args.sticky:
            self.env = gym.make(args.env+'Deterministic-v0').env
        else:
            self.env = envpool.make(f"{args.env}-v5", env_type="gym", num_envs=args.num_envs,episodic_life=False)
            # else: self.env = gym.make(args.env+'Deterministic-v4').env
        self.num_envs=args.num_envs
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        assert type(self.action_space) is gym.spaces.discrete.Discrete
        self.acts_dims = [self.action_space.n]
        self.obs_dims = list(self.observation_space.shape)
        self.render = self.env.render
        
        self.sum_rews=0
        self.steps=0
        self.env_info = {
            'Steps_train': self.steps ,# episode steps
            'Reward_tran': self.sum_rews }

    def get_obs(self):
        return self.last_obs.copy()


    # def get_new_frame(self):
    #     # standard wrapper for atari
    #     frame = self.env._get_obs().astype(np.uint8)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #     frame = cv2.resize(frame, (84,84), interpolation=cv2.INTER_AREA)
    #     self.last_frame = frame.copy()
    #     return frame



    # def get_frame(self):
    #     return self.last_frame.copy()

    # def process_info_steps(self, obs, reward, info):
    #     self.steps += 1
    #     return self.steps

    # def process_info_rewards(self, obs, reward, info):
    #     self.rewards += reward
    #     return self.rewards

    # def process_info(self, obs, reward, info):
    #     return {
    #         remove_color(key): value_func(obs, reward, info)
    #         for key, value_func in self.env_info.items()
        # }

    def rrd_step(self, action):
        self.steps += self.args.num_envs
        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs.copy()
        # if self.steps==self.args.test_timesteps: done = [True for i in range(self.num_envs)]
        return self.last_obs, reward, done, info
    
    def reset(self):
        self.steps = 0
        self.sum_rews = 0.0
        self.step_buffer = {i: 0 for i in range(self.num_envs)}
        self.rews_buffer = {i: [] for i in range(self.num_envs)}  # Reset reward buffer
        self.last_obs = self.env.reset()  # Reset environment and get initial observations
        return self.last_obs.copy()

    # def step(self, action):
    #     obs, reward, done, info = self.env_step(action)
    #     return obs, reward, done, info

    # def reset_ep(self):
    #     self.steps = 0
    #     self.rewards = 0.0

    # def reset(self):
    #     self.reset_ep()
    #     while True:
    #         flag = True
    #         self.env.reset()
    #         for _ in range(max(self.args.noop-self.args.frames,0)):
    #             _, _, done, _ = self.env.step(0)
    #             if done:
    #                 flag = False
    #                 break
    #         if flag: break

    #     self.frames_stack = []
    #     for _ in range(self.args.frames):
    #         self.env.step(0)
    #         self.frames_stack.append(self.get_new_frame())

    #     self.last_obs = np.stack(self.frames_stack, axis=-1)
    #     return self.last_obs.copy()

    
