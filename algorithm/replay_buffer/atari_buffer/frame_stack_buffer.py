import copy
import numpy as np
from torchrl.data.replay_buffers import ReplayBuffer, ListStorage ,TensorStorage,LazyMemmapStorage, TensorDictReplayBuffer
from tensordict import TensorDict
import torch
import numpy as np
from torchrl.data.replay_buffers.samplers import PrioritizedSampler

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


class Episode_FrameStack:
    def __init__(self, info):
        self.common_info = [
            'obs', 'obs_next', 'frame_next',
            'acts', 'rews', 'done'
        ]
        self.frames = info['obs'].shape[-1]
        self.ep = {
            'obs': [],
            'acts': [],
            'rews': [],
            'done': []
        }
        for key in info.keys():
            if not(key in self.common_info):
                self.ep[key] = []
        self.ep_len = 0
        self.sum_rews = 0.0
        self.ep['obs'].append(copy.deepcopy(info['obs']))
    

    def insert(self, info):
        self.ep_len += 1
        self.sum_rews += info['rews']
        self.ep['obs'].append(copy.deepcopy(info['obs_next']))
        self.ep['acts'].append(copy.deepcopy(info['acts']))
        self.ep['rews'].append(copy.deepcopy(info['rews']))
        self.ep['done'].append(copy.deepcopy(info['done']))
        for key in info.keys():
            if not(key in self.common_info):
                self.ep[key].append(copy.deepcopy(info[key]))

    def get_obs(ep,frames, idx):
        idx += 1
        obs = torch.stack(ep['obs'][idx:idx+frames], axis=-1)
        return obs.astype(np.float32)/255.0

    def sample(self):
        idx = np.random.randint(self.ep_len)
        info = {
            'obs': self.get_obs(idx-1),
            'obs_next': self.get_obs(idx),
            'acts': copy.deepcopy(self.ep['acts'][idx]),
            'rews': [copy.deepcopy(self.ep['rews'][idx])],
            'done': [copy.deepcopy(self.ep['done'][idx])]
        }
        for key in self.ep.keys():
            if (not(key in self.common_info)) and (not(key in info.keys())):
                info[key] = copy.deepcopy(self.ep[key][idx])
        return info

    def sample_ircr(self):
        idx = np.random.randint(self.ep_len)
        info = {
            'obs': self.get_obs(idx-1),
            'obs_next': self.get_obs(idx),
            'acts': copy.deepcopy(self.ep['acts'][idx]),
            'rews': [self.sum_rews],  # critical step of IRCR
            'done': [copy.deepcopy(self.ep['done'][idx])]
        }
        for key in self.ep.keys():
            if (not(key in self.common_info)) and (not(key in info.keys())):
                info[key] = copy.deepcopy(self.ep[key][idx])
        return info

    def sample_rrd(self, sample_size, store_coef=False):
        idx = np.random.choice(self.ep_len, sample_size, replace=(sample_size>self.ep_len))
        info = {
            'rrd_obs': [],
            'rrd_obs_next': [],
            'rrd_acts': [],
            'rrd_rews': [self.sum_rews/self.ep_len],
            'rrd_done' :[],

        }
        for _ in range(sample_size):
            idx = np.random.randint(self.ep_len)
            info['rrd_obs'].append(self.get_obs(idx-1))
            info['rrd_obs_next'].append(self.get_obs(idx))
            info['rrd_acts'].append(copy.deepcopy(self.ep['acts'][idx]))
            info['rrd_done'].append(copy.deepcopy(self.ep['done'][idx]))
        if store_coef:
            if (sample_size<=self.ep_len) and (self.ep_len>1):
                info['rrd_var_coef'] = [1.0-float(sample_size)/self.ep_len]
            else:
                info['rrd_var_coef'] = [1.0 if self.ep_len>1 else 0.0]
        return info

class ReplayBuffer_FrameStack:
    def __init__(self, args):
        self.args = args
        self.in_head = True
        self.ep_counter = 0
        self.step_counter = 0
        self.buffer_size = self.args.buffer_size

        self.ep = TensorDictReplayBuffer(storage=LazyMemmapStorage(300,device='cpu'),sampler=PrioritizedSampler(max_capacity=300, alpha=0.8, beta=1.1),priority_key="average_reward",batch_size=1)
        self.length = 0
        self.head_idx = 0
        self.ram_idx = []

        self.sample_batch = {
            'dqn': self.sample_batch_dqn,
            'ircr': self.sample_batch_ircr,
            'rrd_atari_tf': self.sample_batch_rrd,
            'rrd_atari_pytorch': self.sample_batch_rrd,
            'rrd':self.sample_batch_rrd,
        }[args.alg]

    def get_obs(self,ep, idx):
        idx += 1  
        obs = ep[idx]
        return obs.to(dtype=torch.float32) / 255.0  


    def create_tensor_from_list(self,desired_shape, input_list):
        input_list = np.array(input_list)
        tensor = torch.tensor(input_list)

        # Create the result tensor filled with zeros
        result_tensor = torch.zeros(desired_shape, dtype=tensor.dtype)

        # Get the shape of the input tensor
        input_shape = tensor.shape
        
        # Handle cases based on the number of dimensions
        if len(input_shape) == 1:  # Case for 1D input list
            result_tensor[:min(len(input_list), desired_shape[0])] = tensor[:desired_shape[0]]
        else:  # Case for higher-dimensional input
            slices = tuple(slice(0, min(input_shape[i], desired_shape[i])) for i in range(tensor.dim()))
            result_tensor[slices] = tensor[slices]

        result_tensor = result_tensor.to(device)
        return result_tensor
    def store_transition(self, info):
        if self.in_head:
            self.new_ep = Episode_FrameStack(info)
            # self.ep.append(new_ep)
        self.new_ep.insert(info)
        # self.ep[-1].insert(info)
        # self.ram_idx.append(self.ep_counter)

        self.step_counter += 1
        self.in_head = info['done']
        if info['done']:
            tensordict = TensorDict({
            'obs': self.create_tensor_from_list((1500,4,84,84),self.new_ep.ep['obs']),
            'acts': self.create_tensor_from_list((1500,),self.new_ep.ep['acts']),
            'rews': self.create_tensor_from_list((1500,),self.new_ep.ep['rews']),
            'done': self.create_tensor_from_list((1500,),self.new_ep.ep['done']),
            'num_ep':torch.tensor(self.new_ep.ep_len).to(device),
            'sum_rews':torch.tensor(self.new_ep.sum_rews).to(device),
            'average_reward':torch.tensor(self.new_ep.sum_rews/self.new_ep.ep_len).to(device)
            }, batch_size=[],device=device)
            self.ep.add(tensordict)
            self.ep_counter += 1

    def sample_batch_dqn(self, batch_size=-1):
        if batch_size==-1: batch_size = self.args.batch_size
        batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[])

        for i in range(batch_size):
            # idx = self.ram_idx[np.random.randint(self.length)]-self.head_idx
            info = self.ep.sample().sample()
            for key in info.keys():
                batch[key].append(info[key])

        return batch

    def sample_batch_ircr(self, batch_size=-1):
        if batch_size==-1: batch_size = self.args.batch_size
        batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[])

        for i in range(batch_size):
            # idx = self.ram_idx[np.random.randint(self.length)]-self.head_idx
            info = self.ep.sample().sample_ircr()  # critical step of IRCR
            for key in info.keys():
                batch[key].append(info[key])

        return batch

    def sample_batch_rrd(self, batch_size=-1, rrd_batch_size=-1, rrd_sample_size=-1):
        if batch_size==-1: batch_size = self.args.batch_size
        if rrd_batch_size==-1: rrd_batch_size = self.args.rrd_batch_size
        if rrd_sample_size==-1: rrd_sample_size = self.args.rrd_sample_size
        batch = dict( rrd_obs=[], rrd_obs_next=[], rrd_acts=[], rrd_rews=[],rrd_done=[], )
        if self.args.rrd_bias_correction:
            batch['rrd_var_coef'] = []

    
        for i in range(rrd_batch_size//rrd_sample_size):
            # idx = self.ram_idx[np.random.randint(self.length)]-self.head_idx
            # info = self.ep.sample().sample_rrd(rrd_sample_size, store_coef=self.args.rrd_bias_correction)

            while(True):
                episode=self.ep.sample()
                ep_len=episode['num_ep'][0].item()
                if 0<ep_len<1501:
                    break

            info = {
                'rrd_obs': [],
                'rrd_obs_next': [],
                'rrd_acts': [],
                'rrd_rews': [episode['sum_rews'][0]/ep_len],
                'rrd_done' :[],
                }
            for _ in range(rrd_sample_size):
                idx = np.random.randint(ep_len)
                if idx>1501:
                    print(idx,ep_len)
                
                info['rrd_obs'].append(self.get_obs(episode['obs'][0],idx-1))
                info['rrd_obs_next'].append(self.get_obs(episode['obs'][0],idx))
                info['rrd_acts'].append(episode['acts'][0][idx])
                info['rrd_done'].append(episode['done'][0][idx])
            if self.args.rrd_bias_correction:
                if (rrd_sample_size<=ep_len) and (ep_len>1):
                    batch['rrd_var_coef'].append(torch.tensor([1.0-float(rrd_sample_size)/ep_len]))
                else:
                    batch['rrd_var_coef'].append(torch.tensor([1.0 if ep_len>1 else 0.0]))
            for key in info.keys():
                batch[key].append(torch.stack(info[key]))
        batch_tensor=TensorDict({},batch_size=[],device=device)
        for key in batch.keys():
                batch_tensor[key]=torch.stack(batch[key])
        return batch_tensor

