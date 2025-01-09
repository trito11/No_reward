# import copy
# import numpy as np

# from torchrl.data.replay_buffers import ReplayBuffer, ListStorage ,TensorStorage,LazyMemmapStorage, TensorDictReplayBuffer
# from tensordict import TensorDict
# import torch
# import numpy as np
# from torchrl.data.replay_buffers.samplers import PrioritizedSampler

# device = torch.device(
#     "cuda" if torch.cuda.is_available() else
#     "cpu"
# )

# class Trajectory:
#     def __init__(self, init_obs):
#         self.ep = {
#             'obs': [copy.deepcopy(init_obs)],
#             'rews': [],
#             'acts': [],
#             'done': []
#         }
#         self.length = 0
#         self.sum_rews = 0

#     def store_transition(self, info):
#         self.ep['acts'].append(copy.deepcopy(info['acts']))
#         self.ep['obs'].append(copy.deepcopy(info['obs_next']))
#         self.ep['rews'].append(copy.deepcopy(info['rews']))
#         self.ep['done'].append(copy.deepcopy(np.float32(info['done'])))
#         self.length += 1
#         self.sum_rews += info['rews']

#         if info['real_done']:
#             for key in self.ep.keys():
#                 self.ep[key] = np.array(self.ep[key])

#     def sample(self):
#         idx = np.random.randint(self.length)
#         info = {
#             'obs': copy.deepcopy(self.ep['obs'][idx]),
#             'obs_next': copy.deepcopy(self.ep['obs'][idx+1]),
#             'acts': copy.deepcopy(self.ep['acts'][idx]),
#             'rews': [copy.deepcopy(self.ep['rews'][idx])],
#             'done': [copy.deepcopy(self.ep['done'][idx])]
#         }
#         return info

#     def rrd_sample(self, sample_size, store_coef=False):
#         idx = np.random.choice(self.length, sample_size, replace=(sample_size>self.length))
#         info = {
#             'rrd_obs': self.ep['obs'][idx],
#             'rrd_obs_next': self.ep['obs'][idx+1],
#             'rrd_acts': self.ep['acts'][idx],
#             'rrd_rews': [self.sum_rews/self.length]
#         }
#         if store_coef:
#             if (sample_size<=self.length) and (self.length>1):
#                 info['rrd_var_coef'] = [1.0-float(sample_size)/self.length]
#             else:
#                 # We do not handle the case with (sample_size>self.length).
#                 info['rrd_var_coef'] = [1.0 if self.length>1 else 0.0]
#         return info

# class ReplayBuffer_RRD:
#     def __init__(self, args):
#         self.args = args
#         self.ep_counter = 0
#         self.step_counter = 0
#         self.buffer_size = self.args.buffer_size

#         self.ep = TensorDictReplayBuffer(storage=LazyMemmapStorage(300,device='cpu'),batch_size=1)
#         self.ram_idx = []
#         self.length = 0
#         self.head_idx = 0
#         self.in_head = True

#     def store_transition(self, info):
#         if self.in_head:
#             self.new_ep = Trajectory(info['obs'])
#         self.new_ep.store_transition(info)

#         self.step_counter += 1
#         self.in_head = info['real_done']

#         if info['real_done']:
#             tensordict = TensorDict({
#             'obs': self.create_tensor_from_list((2000,111),self.new_ep.ep['obs']),
#             'acts': self.create_tensor_from_list((2000,8),self.new_ep.ep['acts']),
#             'rews': self.create_tensor_from_list((2000,),self.new_ep.ep['rews']),
#             'done': self.create_tensor_from_list((2000,),self.new_ep.ep['done']),
#             'num_ep':torch.tensor(self.new_ep.length).to(device),
#             'sum_rews':torch.tensor(self.new_ep.sum_rews).to(device),
#             'average_reward':torch.tensor(self.new_ep.sum_rews/self.new_ep.length).to(device)
#             }, batch_size=[],device=device)
#             self.ep.add(tensordict)
#             self.ep_counter += 1
#     def create_tensor_from_list(self,desired_shape, input_list):
#         input_list = np.array(input_list)
#         tensor = torch.tensor(input_list)

#         # Create the result tensor filled with zeros
#         result_tensor = torch.zeros(desired_shape, dtype=tensor.dtype)

#         # Get the shape of the input tensor
#         input_shape = tensor.shape
        
#         # Handle cases based on the number of dimensions
#         if len(input_shape) == 1:  # Case for 1D input list
#             result_tensor[:min(len(input_list), desired_shape[0])] = tensor[:desired_shape[0]]
#         else:  # Case for higher-dimensional input
#             slices = tuple(slice(0, min(input_shape[i], desired_shape[i])) for i in range(tensor.dim()))
#             result_tensor[slices] = tensor[slices]

#         result_tensor = result_tensor.to(device)
#         return result_tensor

#     def sample_batch(self, batch_size=-1, rrd_batch_size=-1, rrd_sample_size=-1):
#         if batch_size==-1: batch_size = self.args.batch_size
#         if rrd_batch_size==-1: rrd_batch_size = self.args.rrd_batch_size
#         if rrd_sample_size==-1: rrd_sample_size = self.args.rrd_sample_size
#         batch = dict( rrd_obs=[], rrd_obs_next=[], rrd_acts=[], rrd_rews=[],rrd_done=[], )
#         if self.args.rrd_bias_correction:
#             batch['rrd_var_coef'] = []

#         for i in range(rrd_batch_size//rrd_sample_size):
          
#             while(True):
#                 episode=self.ep.sample()
#                 length=episode['num_ep'][0].item()
#                 if 0<length<1001:
#                     break

#             info = {
#                 'rrd_obs': [],
#                 'rrd_obs_next': [],
#                 'rrd_acts': [],
#                 'rrd_rews': [episode['sum_rews'][0]/length],
#                 'rrd_done' :[],
#                 }
#             for _ in range(rrd_sample_size):
#                 idx = np.random.randint(length)
#                 if idx>1500:
#                     print(idx,length)
                
                
#                 info['rrd_obs'].append(episode['obs'][0][idx])
#                 info['rrd_obs_next'].append(episode['obs'][0][idx+1])
#                 info['rrd_acts'].append(episode['acts'][0][idx])
#                 info['rrd_done'].append(episode['done'][0][idx])
#             if self.args.rrd_bias_correction:
#                 if (rrd_sample_size<=length) and (length>1):
#                     batch['rrd_var_coef'].append(torch.tensor([1.0-float(rrd_sample_size)/length]))
#                 else:
#                     batch['rrd_var_coef'].append(torch.tensor([1.0 if length>1 else 0.0]))
#             for key in info.keys():
#                 batch[key].append(torch.stack(info[key]))
#         batch_tensor=TensorDict({},batch_size=[],device=device)
#         for key in batch.keys():
#                 batch_tensor[key]=torch.stack(batch[key])
#         return batch_tensor

import copy
import numpy as np

class Trajectory:
    def __init__(self, init_obs):
        self.ep = {
            'obs': [copy.deepcopy(init_obs)],
            'rews': [],
            'acts': [],
            'done': []
        }
        self.length = 0
        self.sum_rews = 0

    def store_transition(self, info):
        self.ep['acts'].append(copy.deepcopy(info['acts']))
        self.ep['obs'].append(copy.deepcopy(info['obs_next']))
        self.ep['rews'].append(copy.deepcopy(info['rews']))
        self.ep['done'].append(copy.deepcopy(np.float32(info['done'])))
        self.length += 1
        self.sum_rews += info['rews']

        if info['real_done']:
            for key in self.ep.keys():
                self.ep[key] = np.array(self.ep[key])

    def sample(self):
        idx = np.random.randint(self.length)
        info = {
            'obs': copy.deepcopy(self.ep['obs'][idx]),
            'obs_next': copy.deepcopy(self.ep['obs'][idx+1]),
            'acts': copy.deepcopy(self.ep['acts'][idx]),
            'rews': [copy.deepcopy(self.ep['rews'][idx])],
            'done': [copy.deepcopy(self.ep['done'][idx])]
        }
        return info

    def rrd_sample(self, sample_size, store_coef=False):
        idx = np.random.choice(self.length, sample_size, replace=(sample_size>self.length))
        info = {
            'rrd_obs': self.ep['obs'][idx],
            'rrd_obs_next': self.ep['obs'][idx+1],
            'rrd_acts': self.ep['acts'][idx],
            'rrd_rews': [self.sum_rews/self.length]
        }
        if store_coef:
            if (sample_size<=self.length) and (self.length>1):
                info['rrd_var_coef'] = [1.0-float(sample_size)/self.length]
            else:
                # We do not handle the case with (sample_size>self.length).
                info['rrd_var_coef'] = [1.0 if self.length>1 else 0.0]
        return info

class ReplayBuffer_RRD:
    def __init__(self, args):
        self.args = args
        self.ep_counter = 0
        self.step_counter = 0
        self.buffer_size = self.args.buffer_size

        self.ep = []
        self.ram_idx = []
        self.length = 0
        self.head_idx = 0
        self.in_head = True

    def store_transition(self, info):
        if self.in_head:
            new_ep = Trajectory(info['obs'])
            self.ep.append(new_ep)
        self.ep[-1].store_transition(info)
        self.ram_idx.append(self.ep_counter)
        self.length += 1

        if self.length>self.buffer_size:
            del_len = self.ep[0].length
            self.ep.pop(0)
            self.head_idx += 1
            self.length -= del_len
            self.ram_idx = self.ram_idx[del_len:]

        self.step_counter += 1
        self.in_head = info['real_done']
        if info['real_done']:
            self.ep_counter += 1

    def sample_batch(self, batch_size=-1, rrd_batch_size=-1, rrd_sample_size=-1):
        if batch_size==-1: batch_size = self.args.batch_size
        if rrd_batch_size==-1: rrd_batch_size = self.args.rrd_batch_size
        if rrd_sample_size==-1: rrd_sample_size = self.args.rrd_sample_size
        batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[], rrd_obs=[], rrd_obs_next=[], rrd_acts=[], rrd_rews=[])
        if self.args.rrd_bias_correction:
            batch['rrd_var_coef'] = []

        for i in range(batch_size):
            idx = self.ram_idx[np.random.randint(self.length)]-self.head_idx
            info = self.ep[idx].sample()
            for key in info.keys():
                batch[key].append(info[key])

        for i in range(rrd_batch_size//rrd_sample_size):
            idx = self.ram_idx[np.random.randint(self.length)]-self.head_idx
            info = self.ep[idx].rrd_sample(rrd_sample_size, store_coef=self.args.rrd_bias_correction)
            for key in info.keys():
                batch[key].append(info[key])

        return batch