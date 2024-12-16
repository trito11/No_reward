import numpy as np
import copy 
def create_EpisodicRewardsEnv(basis_env):
    class EpisodicRewardsEnv(basis_env):
        def __init__(self, args):
            super().__init__(args)

        def step(self, action):
            obs, reward, done, info = self.rrd_step(action)
            for i in range(self.args.num_envs):
                self.rews_buffer[i].append(reward[i])
                self.step_buffer[i]+=1
                info=copy.deepcopy(self.step_buffer)
                if self.step_buffer[i]==self.args.test_timesteps: 
                    done[i]=True
            for i in range(self.args.num_envs):
                if done[i]:
                    reward[i]=np.sum(self.rews_buffer[i])
                    self.step_buffer[i]=0
                    self.rews_buffer[i] = []
                else:
                    reward[i]=0.0
            return obs, reward, done, info

    return EpisodicRewardsEnv



