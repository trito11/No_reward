import numpy as np

import torch
# import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

from torch.cuda.amp import GradScaler, autocast
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


scaler = GradScaler('cuda')
class Normalizer:
    def __init__(self, obs_dims, clip=5):
        self.obs_dims = obs_dims
        self.clip = clip
        self.mean = torch.zeros(obs_dims, dtype=torch.float32).to(device)
        self.var = torch.ones(obs_dims, dtype=torch.float32).to(device)
        self.count = 1e-6  # Small value to avoid division by zero

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.size(0)

        # Update the count
        self.count += batch_count

        # Update mean and variance
        new_mean = (self.count * self.mean + batch_count * batch_mean) / self.count
        new_var = (self.count * self.var + batch_count * batch_var + 
                   (self.count * (batch_mean - self.mean)**2) / self.count) / self.count

        self.mean, self.var = new_mean, new_var

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var).clamp(min=self.clip).to(device)


# set up matplotlib
# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(device)

def RRD_atari_pytorch(args):
    class ValueNetwork(nn.Module):
        def __init__(self, arg):
            super(ValueNetwork, self).__init__()
            self.args = args
            self.flatten = False
            self.acts_num = args.acts_dims[0]
            
            if len(self.args.obs_dims) == 1:
                self.mlp_value = self.build_mlp_value().to(device)
            else:
                self.conv_value = self.build_conv_value().to(device)
            self.initialize_weights()
            
        def initialize_weights(self):
            # Initialize weights of the MLP layers
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)


        def build_mlp_value(self):
            return nn.Sequential(
                nn.Linear(self.args.obs_dims[0], 256).to(device),
                nn.ReLU(),
                nn.Linear(256, 256).to(device),
                nn.ReLU(),
                nn.Linear(256, self.acts_num).to(device)
            )

        def build_conv_value(self):
            conv_layers = nn.Sequential(
                nn.Conv2d(self.args.obs_dims[0], 32, kernel_size=8, stride=4, padding=2).to(device),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1).to(device),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1).to(device),
                nn.ReLU()
            )
            return conv_layers

        def forward(self, obs_ph):
            obs_ph = obs_ph.to(device)

            flatten = len(obs_ph.shape) == len(self.args.obs_dims) + 2
            
            if flatten:
                obs_ph = obs_ph.view(-1, *self.args.obs_dims).to(device)

            # obs_ph = obs_ph.permute(0, 3, 1, 2).to(device)

            if len(self.args.obs_dims) == 1:  # MLP path
                q = self.mlp_value(obs_ph)
            else:  # Conv path
                conv_out = self.conv_value(obs_ph)
                conv_out_flat = torch.flatten(conv_out, start_dim=1).to(device)
                
                q_dense_act = nn.Linear(conv_out_flat.shape[1], 512).to(device)(conv_out_flat)
                q_act = nn.Linear(512, self.acts_num).to(device)(F.relu(q_dense_act))

                if self.args.dueling:
                    q_dense_base = nn.Linear(conv_out_flat.shape[1], 512).to(device)(conv_out_flat)
                    q_base = nn.Linear(512, 1).to(device)(F.relu(q_dense_base))
                    q = q_base + q_act - q_act.mean(dim=1, keepdim=True)
                else:
                    q = q_act
            
            if flatten:
                q = q.view(-1, self.args.rrd_sample_size, self.acts_num)
            
            return q

        
        
    class RandomizedReturnDecomposition():
        def __init__(self, args):
            self.args = args
            self.acts_num = args.acts_dims[0]
            self.q_loss = torch.tensor(0.0, dtype=torch.float32).to(device)  # Khởi tạo q_loss
            self.q_pi = torch.tensor(0.0, dtype=torch.float32).to(device)    # Khởi tạo q_pi
            self.q_total_loss = torch.tensor(0.0, dtype=torch.float32).to(device)  # Khởi tạo q_total_loss
            self.r_var = torch.tensor(0.0, dtype=torch.float64).to(device)

            self.train_info_q = {
            'Q_loss': self.q_loss.item()
            }
            self.train_info = { **self.train_info_q }
            self.step_info = {
                'Q_average': self.q_pi.item()
            }
            self.train_info_r = {
                'Q_total_loss': self.q_total_loss.item()
            }

            if args.rrd_bias_correction:
                self.train_info_r['Q_var'] = self.r_var.item()
            self.train_info_q = {**self.train_info_q, **self.train_info_r}
            self.train_info = {**self.train_info, **self.train_info_r}



            self.q_net=ValueNetwork(self.args).to(device)
            self.target_net=ValueNetwork(self.args).to(device)
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.obs_normalizer = Normalizer(self.args.obs_dims)

            if self.args.optimizer == 'adam':
                self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=self.args.q_lr, eps=self.args.Adam_eps, amsgrad=True)
            elif self.args.optimizer == 'rmsprop':
                self.q_optimizer = optim.RMSprop(self.q_net.parameters(), lr=self.args.q_lr, alpha=self.args.RMSProp_decay, eps=self.args.RMSProp_eps)
            
        def mlp_policy(self,Q_values, axis=-1):
            Q_values_max, _ = torch.max(Q_values, dim=axis, keepdim=True)
            Q_values_shifted = Q_values - Q_values_max
            exp_Q = torch.exp(Q_values_shifted)
            sum_exp_Q = torch.sum(exp_Q, dim=axis, keepdim=True)

            action_distribution = exp_Q / sum_exp_Q
            return action_distribution
        
        def normalizer_update(self, batch):
            if self.args.obs_normalization:
                self.obs_normalizer.update(np.concatenate([batch['obs'], batch['obs_next']], axis=0))
        
        def train(self, batch):
            def one_hot(idx):
                idx =   idx.clone().detach()  
                batch_size, sample_size = idx.shape[0], idx.shape[1]
                idx_flat = idx.view(-1)  
                res = torch.zeros(batch_size * sample_size, self.acts_num, device=idx.device)
                res.scatter_(1, idx_flat.unsqueeze(1), 1.0)
                res = res.view(batch_size, sample_size, self.acts_num)
                
                return res
            
             # Chuyển đổi danh sách thành numpy.ndarray
            self.rrd_raw_obs_ph = batch['rrd_obs'].to( dtype=torch.float32).to(device)
             # Chuyển đổi danh sách thành numpy.ndarray
            self.rrd_raw_obs_next_ph = batch['rrd_obs_next'] .to(dtype=torch.float32).to(device)
            self.rrd_acts_ph = batch['rrd_acts'].to(dtype=torch.int64).to(device) if self.args.env_category!='atari' else one_hot(batch['rrd_acts'].to(dtype=torch.int64)).to(device)
            self.rrd_rews_ph = batch['rrd_rews'].to(dtype=torch.float32).to(device)
            self.done_ph = batch['rrd_done'].to(dtype=torch.float32).view(-1,self.args.rrd_sample_size,1).to(device)

            if self.args.rrd_bias_correction:
               self.rrd_var_coef_ph = batch['rrd_var_coef'].clone().detach().to(device)

            if self.args.obs_normalization:

                
                self.obs_ph = self.obs_normalizer.normalize(self.rrd_raw_obs_ph)
                self.obs_next_ph = self.obs_normalizer.normalize(self.rrd_raw_obs_next_ph)
            else:
                self.obs_normalizer = None
                self.obs_ph = self.rrd_raw_obs_ph
                self.obs_next_ph = self.rrd_raw_obs_next_ph

            if self.args.obs_normalization:
                self.rrd_obs_ph = self.obs_normalizer.normalize(self.rrd_raw_obs_ph)
                self.rrd_obs_next_ph = self.obs_normalizer.normalize(self.rrd_raw_obs_next_ph)
            else:
                self.rrd_obs_ph = self.rrd_raw_obs_ph
                self.rrd_obs_next_ph = self.rrd_raw_obs_next_ph

            with autocast('cuda'):
                self.q = self.q_net( self.rrd_obs_ph)
                self.q = self.q.to(device)
                # self.rrd_acts_ph = self.rrd_acts_ph.to(device)
                self.q_action = (self.q * self.rrd_acts_ph).sum(dim=-1, keepdim=True).to(device)
                self.q_pi = self.q.max()
                with torch.no_grad():
                    self.q_t1 = self.target_net(self.rrd_obs_next_ph) #4,32,7
                    self.q_t2 = self.q_net(self.rrd_obs_next_ph)#4,32,7
                if self.args.basis_alg=='dqn':
                    self.best_act=self.q_t2.argmax(dim=-1)#4,32
                    dim1,dim2=self.q_t1.shape[0],self.q_t1.shape[1]
                    self.q_t=self.q_t1[torch.arange(dim1).unsqueeze(1), torch.arange(dim2), self.best_act].unsqueeze(-1)#4,32,1
                if self.args.basis_alg=='sac':
                    self.policy1 = torch.detach(self.mlp_policy(self.q_t2)).to(device)  # Equivalent to tf.stop_gradient
                    self.q_t = self.policy1 * (self.q_t1 - self.args.alpha* self.policy1.log())
                    self.q_t = self.q_t.sum(dim=-1, keepdim=True).detach()
                
                target = (1.0 - self.done_ph) * self.args.gamma * self.q_t  
            
                rrd_rews_pred = self.q_action - target #4,32,1

                
                self.rrd = rrd_rews_pred.mean(dim=1)#4,1
                criterion = nn.SmoothL1Loss()
                self.regularization=self.args.chi2_coeff* torch.square(rrd_rews_pred).mean()
                self.q_loss = criterion(self.rrd_rews_ph, self.rrd)  + self.regularization
                wandb.log({'regularization':self.regularization})


                if self.args.rrd_bias_correction:
                    assert self.args.rrd_sample_size > 1
                    n = self.args.rrd_sample_size
                    
                    # Compute the variance of the predicted rewards
                    rrd_mean = rrd_rews_pred.mean(dim=1,keepdim=True)
                
                    r_var_single = ((rrd_rews_pred - rrd_mean) ** 2).sum(dim=1) / (n - 1)
                    r_var = (r_var_single * self.rrd_var_coef_ph / n).mean()

                    # Calculate the total loss incorporating the variance
                    self.q_total_loss = self.q_loss - r_var
                else:
                    self.q_total_loss = self.q_loss


            self.q_optimizer.zero_grad()
            scaler.scale(self.q_total_loss).backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
            scaler.step(self.q_optimizer)
            scaler.update()

            self.update_target_net()
            
            self.train_info_q = {
            'Q_loss': self.q_loss.item()
            }
            self.train_info = { **self.train_info_q }
            self.step_info = {
                'Q_average': self.q_pi.item()
            }
            self.train_info_r = {
                'Q_total_loss': self.q_total_loss.item()
            }
            if args.rrd_bias_correction:
                self.train_info_r['Q_var'] = r_var.item()
            self.train_info_q = {**self.train_info_q, **self.train_info_r}
            self.train_info = {**self.train_info, **self.train_info_r}
            wandb.log(self.train_info)

            return self.train_info

        def update_target_net(self):
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.q_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * (1 - self.args.polyak) + target_net_state_dict[key] * self.args.polyak
            self.target_net.load_state_dict(target_net_state_dict)




        def step(self, obs, explore=False, test_info=False):
            # Giai đoạn khởi động
            if not test_info and self.args.buffer.step_counter < self.args.warmup:
                return np.random.randint(self.acts_num,size=self.args.num_envs)


            if explore and np.random.uniform() <= self.args.eps_act:
                return np.random.randint(self.acts_num,size=self.args.num_envs)
            obs=obs/255.0
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            # obs_tensor=obs_tensor.unsqueeze(0)
            q = self.q_net(obs_tensor).to(device)
            # q=q.squeeze(0)
            if self.args.basis_alg=='dqn':
                action=q.argmax(dim=-1).cpu().numpy()
            if self.args.basis_alg=='sac':
                policy = self.mlp_policy(q).detach().cpu().numpy()  
                action = [np.random.choice(self.acts_num, p=policy[i]) for i in range(self.args.num_envs)]
                action=np.array(action)
            info = {
                'Q_average': q.max().item()  
            }

            if test_info:
                return action, info
            return action

    return RandomizedReturnDecomposition(args)

