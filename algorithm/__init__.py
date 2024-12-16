from .basis_alg.dqn import DQN
from .basis_alg.ddpg import DDPG
from .basis_alg.td3 import TD3
from .basis_alg.sac import SAC

basis_algorithm_collection = {
    'dqn': DQN,
    'ddpg': DDPG,
    'td3': TD3,
    'sac': SAC
}

from .ircr import IRCR
from .rrd import RRD
from .rrd_ddpg import RRD_ddpg
from .rrd_atari_tf import RRD_atari_tf
from .rrd_atari_pytorch import RRD_atari_pytorch
from .rrd_mujoco_pytorch import RRD_mujoco_pytorch
from .rrd_mujoco_pytorch_xql import RRD_mujoco_pytorch_xql
from .rrd_mujoco_pytorch_q import RRD_mujoco_pytorch_q

advanced_algorithm_collection = {
    'ircr': IRCR,
    'rrd': RRD,
    'rrd_ddpg':RRD_ddpg,
    'rrd_atari_tf':RRD_atari_tf,
    'rrd_atari_pytorch':RRD_atari_pytorch,
    'rrd_mujoco_pytorch':RRD_mujoco_pytorch,
    'rrd_mujoco_pytorch_xql':RRD_mujoco_pytorch_xql,
    'rrd_mujoco_pytorch_q':RRD_mujoco_pytorch_q,

}

algorithm_collection = {
    **basis_algorithm_collection,
    **advanced_algorithm_collection
}

def create_agent(args):
    return algorithm_collection[args.alg](args)

