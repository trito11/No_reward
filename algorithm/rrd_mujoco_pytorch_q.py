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

torch.autograd.set_detect_anomaly(True)
a=123
np.random.seed(a)


random.seed(a)


torch.manual_seed(a)
torch.cuda.manual_seed(a)
scaler = GradScaler(enabled=False)

class RandomNormal:
    def __init__(self, mean, logstd):
        self.raw_logstd = logstd
        if mean.dim() > logstd.dim():
            logstd = mean * 0.0 + logstd  # Broadcasting
        self.mean = mean
        self.logstd = logstd
        self.std = torch.maximum(torch.exp(logstd), torch.tensor(1e-2))

    def log_p(self, x):
        pi_constant = torch.tensor(2.0 * torch.pi, dtype=self.logstd.dtype, device=self.logstd.device)
        return torch.sum(
            -0.5 * torch.log(pi_constant) - self.logstd 
            - 0.5 * torch.square((x - self.mean) / self.std),
            dim=-1, keepdim=True
        )

    def entropy(self):
        return torch.sum(
            self.logstd + 0.5 * torch.log(2.0 * torch.pi * torch.e), 
            dim=-1, keepdim=True
        )

    def kl(self, other):
        return torch.sum(
            -0.5 + other.logstd - self.logstd 
            + 0.5 * torch.square(self.std / other.std) 
            + 0.5 * torch.square((self.mean - other.mean) / other.std),
            dim=-1, keepdim=True
        )
    
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
# Sac-algorithms
def RRD_mujoco_pytorch_q(args):
    class Policy(nn.Module):
        def __init__(self, args):
            super(Policy, self).__init__()  
            self.args = args
            self.acts_num = args.acts_dims[0]

            self.fc1 = nn.Linear(self.args.obs_dims[0], 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc_mean_logstd = nn.Linear(256, self.acts_num * 2)

            # Xavier initialization
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc_mean_logstd.weight)

        def forward(self, obs_ph):
            obs_ph = obs_ph.to(device)
            flatten = len(obs_ph.shape) == len(self.args.obs_dims) + 2
            
            if flatten:
                obs_ph = obs_ph.view(-1, *self.args.obs_dims).to(device)
            x = F.relu(self.fc1(obs_ph))
            x = F.relu(self.fc2(x))
            pi_mean_logstd = self.fc_mean_logstd(x)
            if flatten:
                pi_mean_logstd = pi_mean_logstd.view(-1, self.args.rrd_sample_size, self.acts_num*2)
                pi_mean = pi_mean_logstd[:, :, :self.fc_mean_logstd.out_features // 2]
                pi_logstd = torch.clamp(
                    pi_mean_logstd[:, :, self.fc_mean_logstd.out_features // 2:], 
                    min=-20.0, max=2.0)
            else:
                pi_mean = pi_mean_logstd[:, :self.fc_mean_logstd.out_features // 2]
                pi_logstd = torch.clamp(
                    pi_mean_logstd[:, self.fc_mean_logstd.out_features // 2:], 
                    min=-20.0, max=2.0)
        
            # Split into mean and log std dev, and clip the log std dev
        
            
            return RandomNormal(mean=pi_mean, logstd=pi_logstd)
        
    class QValueNetwork(nn.Module):
        def __init__(self,args):
            super(QValueNetwork, self).__init__()  
            self.args = args
            self.acts_num = args.acts_dims[0]

            self.fc1 = nn.Linear(self.args.obs_dims[0] + self.acts_num, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc_q = nn.Linear(256, 1)

            # Xavier initialization
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc_q.weight)

        def forward(self, obs_ph, acts):
            
            # Concatenate observation and action inputs
            x = torch.cat([obs_ph, acts], dim=-1)
            x = x.to(device).to(torch.float32)

            flatten = len(x.shape) == len(self.args.obs_dims) + 2
            
            if flatten:
                x = x.view(-1, self.args.obs_dims[0]+ self.acts_num).to(device)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            q_value =  self.fc_q(x)
            if flatten:
                q_value=q_value.view(-1, self.args.rrd_sample_size, 1)
            return q_value
        
    class Value(nn.Module):
        def __init__(self,args):
            super(Value, self).__init__() 
            self.args = args
            self.acts_num = args.acts_dims[0]

            self.fc1 = nn.Linear(self.args.obs_dims[0] , 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc_v = nn.Linear(256, 1)

            # Xavier initialization
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc_v.weight)

        def forward(self, obs_ph):
            # Concatenate observation and action inputs
            x = obs_ph.to(device).to(torch.float32)
            flatten = len(obs_ph.shape) == len(self.args.obs_dims) + 2
            if flatten:
               x= x.view(-1, self.args.obs_dims[0]).to(device)
        
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            v_value = self.fc_v(x)
            if flatten:
                v_value=v_value.view(-1, self.args.rrd_sample_size, 1)
            return v_value
        
    class Reward1(nn.Module):
        def __init__(self, args):
            super(Reward1, self).__init__()
            self.args = args
            self.acts_num = args.acts_dims[0]

            # Define a shared network for both current and next state-action pairs
            self.fc1 = nn.Linear(self.args.obs_dims[0] + self.acts_num, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc_q = nn.Linear(256, 1)

            # Xavier initialization
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc_q.weight)

        def forward(self, obs_ph, acts, next_obs_ph, next_acts):
            # Calculate Q-value for current state-action pair
            q_value = self._calculate_q_value(obs_ph, acts)

            # Calculate Q-value for next state-action pair
            q_value_next = self._calculate_q_value(next_obs_ph, next_acts)

            # Reward calculation: r = Q(s, a) - γ * Q(s', a')
            reward = q_value - self.args.gamma * q_value_next

            # Reshape if input was flattened
            if len(obs_ph.shape) == len(self.args.obs_dims) + 2:
                reward = reward.view(-1, self.args.rrd_sample_size, 1)

            return reward

        def _calculate_q_value(self, obs, acts):
            """Helper function to calculate Q-value."""
            # Concatenate observation and action
            x = torch.cat([obs, acts], dim=-1).to(torch.float32)

            # Flatten input if needed
            if len(x.shape) == len(self.args.obs_dims) + 2:
                x = x.view(-1, self.args.obs_dims[0] + self.acts_num)

            # Forward pass through the network
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            q_value = self.fc_q(x)

            return q_value

    class Reward2(nn.Module):
        def __init__(self,args):
            super(Reward2, self).__init__()  
            self.args = args
            self.acts_num = args.acts_dims[0]

            self.fc1 = nn.Linear(2*self.args.obs_dims[0] + self.acts_num, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc_q = nn.Linear(256, 1)

            # Xavier initialization
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc_q.weight)

        def forward(self, obs_ph, acts, next_obs_ph):
            
            # Concatenate observation and action inputs
            x = torch.cat([obs_ph, acts, next_obs_ph], dim=-1)
            x = x.to(device).to(torch.float32)

            flatten = len(x.shape) == len(self.args.obs_dims) + 2
            
            if flatten:
                x = x.view(-1, 2*self.args.obs_dims[0]+ self.acts_num).to(device)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            q_value =  self.fc_q(x)
            if flatten:
                q_value=q_value.view(-1, self.args.rrd_sample_size, 1)
            return q_value

    class Reward3(nn.Module):
        def __init__(self, args):
            super(Reward3, self).__init__()  
            self.args = args
            self.acts_num = args.acts_dims[0]
            obs_act_dim = self.args.obs_dims[0] + self.acts_num

            # Q-network layers
            self.fc1 = nn.Linear(obs_act_dim, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc_q = nn.Linear(256, 1)

            # Target Q-network layers
            self.fc1_t = nn.Linear(obs_act_dim, 256)
            self.fc2_t = nn.Linear(256, 256)
            self.fc_q_t = nn.Linear(256, 1)

            # Xavier initialization
            for layer in [self.fc1, self.fc2, self.fc_q, self.fc1_t, self.fc2_t, self.fc_q_t]:
                nn.init.xavier_uniform_(layer.weight)

        def forward(self, obs_ph, acts, next_obs_ph, next_acts):
            # Concatenate observations and actions
            x1 = torch.cat([obs_ph, acts], dim=-1).float()
            x2 = torch.cat([next_obs_ph, next_acts], dim=-1).float()

            # Flatten if needed
            flatten = len(x1.shape) == len(self.args.obs_dims) + 2
            if flatten:
                x1 = x1.view(-1, self.args.obs_dims[0] + self.acts_num)
                x2 = x2.view(-1, self.args.obs_dims[0] + self.acts_num)

            # Compute Q-values for current and next states
            q_value = self._compute_q_value(x1, self.fc1, self.fc2, self.fc_q)
            q_value_t = self._compute_q_value(x2, self.fc1_t, self.fc2_t, self.fc_q_t)

            # Calculate reward: R(s, a) = Q(s, a) - γ * Q(s', a')
            reward = q_value - self.args.gamma * q_value_t

            # Reshape back if flattened
            if flatten:
                reward = reward.view(-1, self.args.rrd_sample_size, 1)
            return reward

        def _compute_q_value(self, x, fc1, fc2, fc_q):
            """Helper function to compute Q-value."""
            x = F.relu(fc1(x))
            x = F.relu(fc2(x))
            q_value = fc_q(x)
            return q_value
    
    class AlphaUpdater():

        def __init__(self, alpha_init, alpha_lr, acts_dims):
            # Initialize log_alpha as a trainable parameter
            self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(alpha_init), dtype=torch.float32))
            self.alpha = lambda: torch.exp(self.log_alpha)  # Alpha is derived from log_alpha

            # Store other parameters
            self.alpha_lr = alpha_lr
            self.acts_dims = acts_dims

            # Define optimizer
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)

        def compute_alpha_loss(self, pi_log_p):
            
            target_entropy = np.prod(self.acts_dims)  # Similar to np.prod(args.acts_dims)
            alpha_loss = -self.log_alpha * (pi_log_p.mean().detach() - target_entropy)
            return alpha_loss

        def update_alpha(self, pi_log_p):
           
            alpha_loss = self.compute_alpha_loss(pi_log_p)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()


    class RandomizedReturnDecomposition():
        def __init__(self, args):
            self.args = args
            self.acts_num = args.acts_dims[0]
            self.q_loss = torch.tensor(0.0, dtype=torch.float32).to(device)  # Khởi tạo q_loss
            self.q_pi = torch.tensor(0.0, dtype=torch.float32).to(device)    # Khởi tạo q_pi
            self.q_total_loss = torch.tensor(0.0, dtype=torch.float32).to(device)  # Khởi tạo q_total_loss
            self.r_var = torch.tensor(0.0, dtype=torch.float64).to(device)
            self.r_loss = torch.tensor(0.0, dtype=torch.float64).to(device)
            self.alpha=self.args.alpha
            self.target_clipping=True
            self.clip_score=10
            self.beta=1

            self.train_info_q = {
            'Q_loss': self.q_loss.item()
            }
            self.train_info = { **self.train_info_q }
            self.step_info = {
                'Q_average': torch.mean(self.q_pi).item(),
                # 'Pi_step_std': self.q_loss
            }
            self.train_info_r = {
                # 'R_loss':self.r_loss.item(),
                'Q_total_loss': self.q_total_loss.item()
            }

            if args.rrd_bias_correction:
                self.train_info_r['Q_var'] = self.r_var.item()
            self.train_info_q = {**self.train_info_q, **self.train_info_r}
            self.train_info = {**self.train_info, **self.train_info_r}



            self.q_net1=QValueNetwork(self.args).to(device)
            self.target_net1=QValueNetwork(self.args).to(device)
            self.target_net1.load_state_dict(self.q_net1.state_dict())
            for param in self.target_net1.parameters():
                param.requires_grad = False

            self.q_net2=QValueNetwork(self.args).to(device)
            self.target_net2=QValueNetwork(self.args).to(device)
            self.target_net2.load_state_dict(self.q_net1.state_dict())
            for param in self.target_net2.parameters():
                param.requires_grad = False

            self.value=Value(self.args).to(device)

       
            self.policy=Policy(self.args).to(device)

            self.obs_normalizer = Normalizer(self.args.obs_dims)
            
            self.reward1=Reward1(self.args).to(device)
            self.reward2=Reward2(self.args).to(device)
            self.reward3=Reward3(self.args).to(device)

            if self.args.optimizer == 'adam':
                self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=self.args.q_lr, eps=self.args.Adam_eps, amsgrad=True)
                self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=self.args.q_lr, eps=self.args.Adam_eps, amsgrad=True)
                self.reward_optimizer1=optim.Adam(self.reward1.parameters(), lr=self.args.q_lr, eps=self.args.Adam_eps, amsgrad=True)
                self.reward_optimizer2=optim.Adam(self.reward2.parameters(), lr=self.args.q_lr, eps=self.args.Adam_eps, amsgrad=True)
                self.reward_optimizer3=optim.Adam(self.reward3.parameters(), lr=self.args.q_lr, eps=self.args.Adam_eps, amsgrad=True)
                self.v_optimizer  = optim.Adam(self.value.parameters(), lr=1e-4, eps=self.args.Adam_eps, amsgrad=True)
                self.pi_optimizer = optim.Adam(self.policy.parameters(), lr=self.args.pi_lr) 
   


            elif self.args.optimizer == 'rmsprop':
                self.q_optimizer = optim.RMSprop(self.q_net.parameters(), lr=self.args.q_lr, alpha=self.args.RMSProp_decay, eps=self.args.RMSProp_eps)

            self.alpha_update=AlphaUpdater(self.args.alpha_init, self.args.alpha_lr,self.args.acts_dims)

        def get_pi_log_p(self, pi, pi_sample, pi_act):
            log_p_sample = pi.log_p(pi_sample)  # Log-probability of pi_sample
            eps = 1e-6  # Small value to ensure numerical stability
            adjustment = torch.sum(torch.log(1 - pi_act.pow(2)+eps), dim=-1, keepdim=True)  # Adjustment term
            return log_p_sample - adjustment
        
        def normalizer_update(self, batch):
            if self.args.obs_normalization:
                self.obs_normalizer.update(np.concatenate([batch['obs'], batch['obs_next']], axis=0))
        
        
        def train(self, batch,train_q=False,train_policy=False):


            def one_hot(idx):
                idx =   idx.clone().detach()  
                batch_size, sample_size = idx.shape[0], idx.shape[1]
                idx_flat = idx.view(-1)  
                res = torch.zeros(batch_size * sample_size, self.acts_num, device=idx.device)
                res.scatter_(1, idx_flat.unsqueeze(1), 1.0)
                res = res.view(batch_size, sample_size, self.acts_num)
                
                return res
            
       
            self.rrd_raw_obs_ph = batch['rrd_obs'].to( dtype=torch.float32).to(device)
    
            self.rrd_raw_obs_next_ph = batch['rrd_obs_next'] .to(dtype=torch.float32).to(device)
            self.rrd_acts_ph = batch['rrd_acts'].to(dtype=torch.float32).to(device) if self.args.env_category!='atari' else one_hot(batch['rrd_acts'].to(dtype=torch.int64)).to(device)
            self.rrd_rews_ph = batch['rrd_rews'].to(dtype=torch.float32).to(device)
            self.done_ph = batch['rrd_done'].to(dtype=torch.float32).view(-1,self.args.rrd_sample_size,1).to(device)
            noise_size = self.rrd_acts_ph.shape
            self.pi_noise_ph=np.random.normal(0.0, 1.0, size=noise_size)
            self.pi_next_noise_ph=np.random.normal(0.0, 1.0, size=noise_size)
            self.pi_noise_ph=torch.tensor(self.pi_noise_ph).to(device)
            self.pi_next_noise_ph=torch.tensor(self.pi_next_noise_ph).to(device)


        
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




            with autocast(enabled=False):
         

                self.pi = self.policy(self.obs_ph)
                self.pi_sample = self.pi.mean+self.pi_noise_ph*self.pi.std
                self.pi_act = torch.tanh(self.pi_sample)

                self.pi_log_p = self.get_pi_log_p(self.pi, self.pi_sample, self.pi_act)
                self.pi_log_p = torch.nan_to_num(self.pi_log_p, nan=-1e9, posinf=-1e9, neginf=-1e9)
    
                self.q_pi_1 = self.target_net2(self.rrd_obs_ph, self.pi_act)
                # self.q_pi_2=self.q_net2( self.rrd_obs_ph,self.pi_act)         
                
                self.q_pi = self.q_pi_1 - self.alpha * self.pi_log_p
   
                self.q_pi= self.q_pi.to(device)


                with torch.no_grad():
                    self.pi_next = self.policy(self.obs_next_ph)
                # self.next_value=self.v(self.obs_next_ph)
                    self.pi_next_sample = self.pi_next.mean+self.pi_next_noise_ph*self.pi_next.std
                    self.pi_next_act = torch.tanh(self.pi_next_sample)
                    self.pi_next_log_p = self.get_pi_log_p(self.pi_next, self.pi_next_sample, self.pi_next_act)

                self.q_t_1=self.q_net1(self.obs_next_ph, self.pi_next_act)
                self.q_t_2=self.target_net1(self.obs_next_ph, self.pi_next_act)


                self.q_t_2=  self.q_t_2 - self.alpha * self.pi_next_log_p
                # self.q_t_1= self.q_t_1 - self.alpha * self.pi_next_log_p
                self.q_t_1= self.q_t_1.to(device)
                
                criterion = nn.SmoothL1Loss()
                criterion1 = nn.MSELoss()

                if self.args.alpha_lr>0 :
                    self.alpha_update.update_alpha(self.pi_log_p)
                    self.alpha=torch.exp(self.alpha_update.log_alpha).detach()
                    wandb.log({'Alpha':self.alpha})
                

            
                target = (1.0 - self.done_ph) * self.args.gamma * self.q_t_2
                self.q_1=self.q_net1( self.obs_ph,self.rrd_acts_ph) 
                rrd_rews_pred = self.q_1 - target 
                self.rrd = rrd_rews_pred.to(torch.float32).mean(dim=1)


                # reward=self.reward1(self.obs_ph,self.rrd_acts_ph,self.obs_next_ph,self.pi_next_act)
                # self.e_reward1=reward.mean(dim=1).to(torch.float32)

                # self.e_reward2=self.reward2(self.obs_ph,self.rrd_acts_ph,self.obs_next_ph).mean(dim=1).to(torch.float32)
                # self.e_reward3=self.reward3(self.obs_ph,self.rrd_acts_ph,self.obs_next_ph,self.pi_next_act).mean(dim=1).to(torch.float32)

                # self.reward_loss1=criterion1(self.rrd_rews_ph, self.e_reward1)  
                # self.reward_loss2=criterion1(self.rrd_rews_ph, self.e_reward2)  
                # self.reward_loss3=criterion1(self.rrd_rews_ph, self.e_reward3)  

          

                self.regularization= self.args.chi2_coeff* torch.square(rrd_rews_pred).mean()/10
                self.regularization2= self.args.chi2_coeff1*criterion(self.q_t_2, self.q_t_1)

                self.q_loss = criterion(self.rrd_rews_ph, self.rrd)  
                self.q_loss_re = self.q_loss + self.regularization + self.regularization2

                # self.r_loss=criterion1(self.e_reward1,self.rrd)

                
                wandb.log({'regularization':self.regularization})
                wandb.log({'regularization2':self.regularization2})


                if self.args.rrd_bias_correction:
                    assert self.args.rrd_sample_size > 1
                    n = self.args.rrd_sample_size
                    
                    # Compute the variance of the predicted rewards
                    rrd_mean = rrd_rews_pred.mean(dim=1,keepdim=True)
                
                    r_var_single = ((rrd_rews_pred - rrd_mean) ** 2).sum(dim=1) / (n - 1)
                    r_var = (r_var_single * self.rrd_var_coef_ph / n).mean()

                    # Calculate the total loss incorporating the variance
                    self.q_total_loss = self.q_loss_re - r_var
                else:
                    self.q_total_loss = self.q_loss_re           


                if train_policy:
                    self.pi_loss = torch.mean(-self.q_pi) 
                    wandb.log({'Pi_loss':self.pi_loss})
                    
                    self.pi_optimizer.zero_grad()
                    scaler.scale(self.pi_loss).backward()
                    # In-place gradient clipping
                    torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
                    scaler.step(self.pi_optimizer)
                    scaler.update()


                if train_q:
    
                    self.q_optimizer1.zero_grad()
                    scaler.scale(self.q_total_loss).backward()
                    torch.nn.utils.clip_grad_value_(self.q_net1.parameters(), 10)
                    scaler.step(self.q_optimizer1)
                    scaler.update()



                    # self.reward_optimizer1.zero_grad()
                    # scaler.scale(self.reward_loss1).backward()
                    # torch.nn.utils.clip_grad_value_(self.reward1.parameters(), 10)
                    # scaler.step(self.reward_optimizer1)
                    # scaler.update()

                    # reward_state_dict = self.reward1.state_dict()
                    # self.q_net1.load_state_dict(reward_state_dict)

                    # self.reward_optimizer2.zero_grad()
                    # scaler.scale(self.reward_loss2).backward()
                    # torch.nn.utils.clip_grad_value_(self.reward2.parameters(), 10)
                    # scaler.step(self.reward_optimizer2)
                    # scaler.update()

                    # self.reward_optimizer3.zero_grad()
                    # scaler.scale(self.reward_loss3).backward()
                    # torch.nn.utils.clip_grad_value_(self.reward3.parameters(), 10)
                    # scaler.step(self.reward_optimizer3)
                    # scaler.update()
                    # wandb.log({"Reward_loss1":self.reward_loss1})
                    # wandb.log({"Reward_loss2":self.reward_loss2})
                    # wandb.log({"Reward_loss3":self.reward_loss3})


                # self.update_target_net()


            self.train_info_q = {
            'Q_loss': self.q_loss.item()
            }
            self.train_info = { **self.train_info_q }
            self.step_info = {
                'Q_average': torch.mean(self.q_pi).item()
            }
            self.train_info_r = {
                # 'R_loss':self.r_loss.item(),
                'Q_total_loss': self.q_total_loss.item()
            }
            if args.rrd_bias_correction:
                self.train_info_r['Q_var'] = r_var.item()
            self.train_info_q = {**self.train_info_q, **self.train_info_r}
            self.train_info = {**self.train_info, **self.train_info_r}
            wandb.log(self.train_info)

            return self.train_info

        def update_target_net(self):




            target_net1_state_dict = self.target_net1.state_dict()
            policy_net1_state_dict = self.q_net1.state_dict()
            for key in policy_net1_state_dict:
                target_net1_state_dict[key] = policy_net1_state_dict[key] * (1 - self.args.polyak) + target_net1_state_dict[key] * self.args.polyak
            self.target_net1.load_state_dict(target_net1_state_dict)
            self.target_net2.load_state_dict(policy_net1_state_dict)




        def step(self, obs, explore=False, test_info=False):
            # Giai đoạn khởi động
            if explore:
                noise= np.random.normal(0.0, 1.0,size=[self.args.num_envs,self.acts_num])
            else:
                noise = np.zeros(shape=[self.args.num_envs,self.acts_num], dtype=np.float32)
            
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            noise=torch.tensor(noise).to(device)
            self.pi = self.policy(obs_tensor)
            self.pi_sample = self.pi.mean+noise*self.pi.std
            wandb.log({'Pi_std':self.pi.std.mean()})
            self.pi_act = torch.tanh(self.pi_sample)
            action= self.pi_act.detach().cpu().numpy().astype(np.float64)
            self.pi_log_p = self.get_pi_log_p(self.pi, self.pi_sample, self.pi_act)
            self.q_pi_1=self.q_net1( obs_tensor,self.pi_act)
            self.q_pi = self.q_pi_1- self.alpha * self.pi_log_p
            self.q_pi= self.q_pi.to(device)

            
            info = {
                'Q_average': torch.mean(self.q_pi_1).item(),
                # 'Pi_step_std': torch.mean(self.pi.std).item()
            }
            wandb.log(info)

            if test_info:
                return action, info
            return action

    return RandomizedReturnDecomposition(args)


