import gym
import numpy as np
# from utils.os_utils_tf import remove_color
import envpool

class MuJoCoNormalEnv():
    def __init__(self, args):
        self.args = args
        # When using Ant-v2
        self.env = gym.make(args.env).env
        # When using Ant-v3-multi env
        # self.env = envpool.make(args.env, num_envs=args.num_envs,env_type="gym")

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.render = self.env.render
        self.acts_dims = list(self.action_space.shape)
        self.obs_dims = list(self.observation_space.shape)

        self.action_scale = np.array(self.action_space.high)
        for value_low, value_high in zip(list(self.action_space.low), list(self.action_space.high)):
            assert abs(value_low+value_high)<1e-3, (value_low, value_high)

        self.sum_rews=0
        self.steps=0
        self.env_info = {
            'Steps_train': self.steps ,# episode steps
            'Reward_train': self.sum_rews }
        
    def get_obs(self):
        return self.last_obs.copy()

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

    # def env_step(self, action):
    #     obs, reward, done, info = self.env.step(action*self.action_scale)
    #     info = self.process_info(obs, reward, info)
    #     self.last_obs = obs.copy()
    #     if self.steps==self.args.test_timesteps: done = True
    #     return obs, reward, done, info

    # def step(self, action):
    #     obs, reward, done, info = self.env_step(action)
    #     return obs, reward, done, info

    # When using Ant-v3
    # def rrd_step(self, action):
    #     self.steps += self.args.num_envs
    #     obs, reward, done, info,_ = self.env.step(action)
    #     self.last_obs = obs.copy()
    

    #     return self.last_obs, reward, done, info
# When using Ant-v2
    def rrd_step(self, action):
        self.steps += self.args.num_envs
        obs, reward, done, info ,_= self.env.step(action[0])
        obs=np.expand_dims(obs, axis=0)
        reward=np.expand_dims(reward, axis=0)
        done=np.expand_dims(done, axis=0)
        self.last_obs = obs.copy()
    

        return self.last_obs, reward, done, info
    
    # def reset_ep(self):
    #     self.steps = 0
    #     self.rewards = 0.0
    #     self.last_obs = (self.env.reset()).copy()

    # def reset(self):
    #     self.reset_ep()
    #     return self.last_obs.copy()
# When using Ant-v3
    # def reset(self):
    #     self.steps=0

    #     self.sum_rews = 0.0
    #     self.rews_buffer = {i: [] for i in range(self.args.num_envs)}  # Reset reward buffer
    #     self.step_buffer = {i: 0 for i in range(self.args.num_envs)}

    #     self.last_obs = self.env.reset()  # Reset environment and get initial observations
    #     return self.last_obs[0].copy()
    
# When using Ant-v2
    def reset(self):
        self.steps=0

        self.sum_rews = 0.0
        self.rews_buffer = {i: [] for i in range(self.args.num_envs)}  # Reset reward buffer
        self.step_buffer = {i: 0 for i in range(self.args.num_envs)}

        self.last_obs = self.env.reset()  # Reset environment and get initial observations
        return np.expand_dims(self.last_obs[0].copy(), axis=0)
    