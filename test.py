import numpy as np
from envs import make_env
from utils.os_utils_tf import make_dir
import wandb
class Tester:
    def __init__(self, args):
        self.args = args
        self.env = make_env(args)
        self.info = []

        if args.save_rews:
            make_dir('log/rews', clear=False)
            self.rews_record = {}
            self.rews_record[args.env] = []

    def test_rollouts(self):
        rewards_sum = 0.0
        step_sum=0
        epi=0
        run=0
        done_flags = [False] * self.args.num_envs
        for _ in range(1):
            obs = self.env.reset()
            for timestep in range(self.args.test_timesteps):
                action, info = self.args.agent.step(obs, explore=False, test_info=True)
                # wandb.log(info)
                self.args.logger.add_dict(info)
                if 'test_eps' in self.args.__dict__.keys():
                    # the default testing scheme of Atari games
                    if np.random.uniform(0.0, 1.0)<=self.args.test_eps:
                        action = np.random.randint(self.env.acts_dims,size=self.args.num_envs)
                obs, reward, done, info = self.env.step(action)
                for i in range(self.args.num_envs):
            
                    if done[i]:
                        epi+=1
                        rewards_sum += reward[i]
                        step_sum+=info[i]
                        # print(info[i])
                        self.args.logger.add_dict({'Steps_train':info[i],'Reward_train':reward[i]})
                        done_flags[i]='True'
                        # print(reward[i])

                        wandb.log({'Rewards':reward[i]})
                        wandb.log({'Step':info[i]})


                if epi==5:break

        if self.args.save_rews:
            step = self.args.learner.step_counter
            step_aver=step_sum/epi
            rews = rewards_sum/epi
            wandb.log({'Average_Rewards':rews})
            wandb.log({'Average_Step':step_aver})

            self.rews_record[self.args.env].append((step, rews))
            

    def cycle_summary(self):
        self.test_rollouts()

    def epoch_summary(self):
        if self.args.save_rews:
            for key, acc_info in self.rews_record.items():
                log_folder = 'rews'
                if self.args.tag!='': log_folder = log_folder+'/'+self.args.tag+'/'+self.args.tag
                self.args.logger.save_npz(acc_info, key, log_folder)

    def final_summary(self):
        pass

