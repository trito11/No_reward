# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from buffers import TrajectoryReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 3000000
    """total timesteps of the experiments"""
    buffer_size: int = int(100)
    """the replay memory buffer size"""
    trajectory_size: int = int(1e4)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    q_coefficient: float = 0

    var_coefficient: float = 1

    max_eps_len:int = 100

    n_eps: int = 4 
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5
# Sua buffer return trajectory, 
class Reward(nn.Module):
    def __init__(self, env):
        super().__init__()  
        obs_dim=np.array(env.single_observation_space.shape).prod()
        act_dim=np.prod(env.single_action_space.shape)
        
        self.fc1 = nn.Linear(2*obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_q = nn.Linear(256, 1)

    def forward(self, obs_ph, acts, next_obs_ph):
        x = torch.cat([obs_ph, acts, next_obs_ph], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value =  self.fc_q(x)
        return q_value
    
class Reward_1(nn.Module):
    def __init__(self, env):
        super().__init__()  
        obs_dim=np.array(env.single_observation_space.shape).prod()
        act_dim=np.prod(env.single_action_space.shape)
        
        self.fc1 = nn.Linear(2*obs_dim + 2*act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_q = nn.Linear(256, 1)

    def forward(self, obs_ph, acts, next_obs_ph, next_acts):
        x = torch.cat([obs_ph, acts, next_obs_ph, next_acts], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value =  self.fc_q(x)
        return q_value


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,

            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    reward_net = Reward(envs).to(device)
    reward_test = Reward_1(envs).to(device)

    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    re_optimizer = optim.Adam(list(reward_net.parameters()), lr=args.policy_lr)
    re_test_optimizer = optim.Adam(list(reward_test.parameters()), lr=args.policy_lr)


    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = TrajectoryReplayBuffer(
        args.trajectory_size,
        args.buffer_size,
        args.max_eps_len,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            # print(infos["final_info"])
            # exit()
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']},  episodic_length={info['episode']['l']} ")

                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        # print(terminations.shape)
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = []
            qf_loss = 0
            q_loss = 0
            q_r_loss = 0 
            for i in range(args.n_eps):
                data.append(rb.sample(args.batch_size))
            
                mb_rewards = data[i].rewards
                mb_obs = data[i].observations
                mb_act = data[i].actions
                mb_obs_next = data[i].next_observations
                mb_dones = data[i].dones
                mb_eps_rewards=data[i].eps_rewards
            # print(mb_rewards.shape)
            # print(mb_obs.shape)
            # print(mb_act.shape)
            # print(mb_obs_next.shape)
            # print(mb_dones.shape)
            # print(mb_eps_rewards)
            # print(mb_eps_rewards.shape)
            # exit()


                pre_rewards = reward_net(mb_obs, mb_act, mb_obs_next)

                # print(pre_rewards.shape)

                re_loss = F.mse_loss(torch.mean(pre_rewards), torch.mean(mb_eps_rewards))
                # print(loss_re)

                re_optimizer.zero_grad()
                re_loss.backward()
                re_optimizer.step()

                next_state_actions, next_state_log_pi, _ = actor.get_action(mb_obs_next)

                reward_2 = reward_test(mb_obs, mb_act, mb_obs_next, next_state_actions)
                re_test_loss = F.mse_loss(torch.mean(reward_2), torch.mean(mb_eps_rewards))
                # print(loss_re
                re_test_optimizer.zero_grad()
                re_test_loss.backward()
                re_test_optimizer.step()

                diff_re = F.mse_loss(pre_rewards, reward_2).mean()





                
                with torch.no_grad():

                    pre_rewards = reward_net(mb_obs, mb_act, mb_obs_next)

                    next_state_actions, next_state_log_pi, _ = actor.get_action(mb_obs_next)
                    qf1_next_target = qf1_target(mb_obs_next, next_state_actions)
                    qf2_next_target = qf2_target(mb_obs_next, next_state_actions)

                    # qf1_next_target = qf1(data.next_observations, next_state_actions)
                    # qf2_next_target = qf2(data.next_observations, next_state_actions)

                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_v_value = (1 - mb_dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                    # next_q_value = pre_rewards.flatten() + (1 - mb_dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                    next_q_value = mb_rewards.flatten() + (1 - mb_dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)


        
                qf1_a_values = qf1(mb_obs, mb_act).view(-1)
                qf2_a_values = qf2(mb_obs, mb_act).view(-1)

                
                # pre_re_1 = qf1_a_values - next_v_value
                # pre_re_2 = qf2_a_values - next_v_value
                # next_v_value1 = (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)
                # next_v_value2 = (1 - data.dones.flatten()) * args.gamma * (qf2_next_target).view(-1)





                pre_re_1 = qf1_a_values - next_v_value
                pre_re_2 = qf2_a_values - next_v_value

                # a1= qf1_a_values - qf2_a_values.detach()
                # a2= qf2_a_values - qf1_a_values.detach()

                a1= pre_re_1 - mb_rewards.flatten()
                a2= pre_re_2 - mb_rewards.flatten()


                n = args.batch_size

                r_mean1 = pre_re_1.mean()
                r_var_single1 = ((pre_re_1 - r_mean1) ** 2).sum() / (n - 1)
                r_var1 = (r_var_single1 ).mean()

                r_mean2 = pre_re_2.mean()
                r_var_single2 = ((pre_re_2 - r_mean2) ** 2).sum() / (n - 1)
                r_var2 = (r_var_single2 ).mean()

                a_mean1 = a1.mean()
                a_var_single1 = ((a1 - a_mean1) ** 2).sum() / (n - 1)
                a_var1 = (a_var_single1 ).mean()

                a_mean2 = a2.mean()
                a_var_single2 = ((a2 - a_mean2) ** 2).sum() / (n - 1)
                a_var2 = (a_var_single2 ).mean()


                r_mean = mb_rewards.mean()
                r_var_single = ((mb_rewards - r_mean) ** 2).sum() / (n - 1)
                r_var = (r_var_single ).mean()
    

                # qf1_loss = F.mse_loss(torch.mean(mb_eps_rewards), torch.mean(pre_re_1)) - r_var1 - a_var1
                # qf2_loss = F.mse_loss(torch.mean(mb_eps_rewards), torch.mean(pre_re_2)) - r_var2 - a_var2

                qf1_loss = F.mse_loss(torch.mean(mb_eps_rewards), torch.mean(pre_re_1)) + 0.5 * torch.square(pre_re_1 - torch.mean(mb_eps_rewards)).mean() + args.var_coefficient * r_var1 + args.q_coefficient * torch.mean( 1/(2* torch.mean(mb_eps_rewards))  * pre_re_1**2-pre_re_1) + 2 * pre_re_1.mean()*mb_eps_rewards.mean() - 2 * (pre_re_1 * torch.mean(mb_eps_rewards)).mean()
                qf2_loss = F.mse_loss(torch.mean(mb_eps_rewards), torch.mean(pre_re_2)) + 0.5 * torch.square(pre_re_2 - torch.mean(mb_eps_rewards)).mean() + args.var_coefficient * r_var2 + args.q_coefficient * torch.mean( 1/(2 * torch.mean(mb_eps_rewards)) * pre_re_2**2-pre_re_2) + 2 * pre_re_2.mean()*mb_eps_rewards.mean() - 2 * (pre_re_2 * torch.mean(mb_eps_rewards)).mean()

                # qf1_loss = F.mse_loss(torch.mean(mb_eps_rewards), torch.mean(pre_re_1)) + 0.5 * torch.square(pre_re_1 - torch.mean(mb_eps_rewards)).mean() + args.var_coefficient * r_var1 + args.q_coefficient * torch.mean( 1/(2* torch.mean(mb_eps_rewards))  * pre_re_1**2-pre_re_1) 
                # qf2_loss = F.mse_loss(torch.mean(mb_eps_rewards), torch.mean(pre_re_2)) + 0.5 * torch.square(pre_re_2 - torch.mean(mb_eps_rewards)).mean() + args.var_coefficient * r_var2 + args.q_coefficient * torch.mean( 1/(2 * torch.mean(mb_eps_rewards)) * pre_re_2**2-pre_re_2) 

                
                q1_loss = F.mse_loss(pre_re_1, mb_rewards.flatten())
                q2_loss = F.mse_loss(pre_re_2, mb_rewards.flatten())


                qf_loss += qf1_loss + qf2_loss
                q_loss += q1_loss + q2_loss

                pre_re = torch.min(pre_re_1, pre_re_2)

                q_r_loss += F.mse_loss(pre_re, pre_rewards.flatten())

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()


            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    actor_loss = 0
                    for i in range(args.n_eps):
                        pi, log_pi, _ = actor.get_action(data[i].observations)
                        qf1_pi = qf1(data[i].observations, pi)
                        qf2_pi = qf2(data[i].observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss += ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        alpha_loss = 0
                        for i in range(args.n_eps):
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(data[i].observations)
                            alpha_loss += (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0/ args.n_eps, global_step)
                writer.add_scalar("losses/q_loss", q_loss.item() / 2.0/ args.n_eps, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("losses/re_loss", re_loss.item(), global_step)
                writer.add_scalar("losses/re_test_loss", re_test_loss.item(), global_step)
                writer.add_scalar("losses/diff_re", diff_re.item(), global_step)


                writer.add_scalar("losses/q_r_loss", q_r_loss.item()/ args.n_eps, global_step)
                writer.add_scalar("rewards/pre_re", pre_re.mean().item(), global_step)
                writer.add_scalar("rewards/pre_rewards", pre_rewards.mean().item(), global_step)
                writer.add_scalar("rewards/eps_rewards", mb_eps_rewards.mean().item(), global_step)
                writer.add_scalar("rewards/batch_rewards", mb_rewards.mean().item(), global_step)
                writer.add_scalar("rewards/true_r_var", r_var.mean().item(), global_step)
                writer.add_scalar("rewards/a_var1", a_var1.mean().item(), global_step)
                writer.add_scalar("rewards/a_var2", a_var2.mean().item(), global_step)
                writer.add_scalar("rewards/r_var1", r_var1.mean().item(), global_step)
                writer.add_scalar("rewards/r_var2", r_var2.mean().item(), global_step)
                writer.add_scalar("rewards/r_q_v_max", pre_re.max().item(), global_step)
                writer.add_scalar("rewards/r_q_v_min", pre_re.min().item(), global_step)
                writer.add_scalar("rewards/r_true_max", mb_rewards.flatten().max().item(), global_step)
                writer.add_scalar("rewards/r_true_min", mb_rewards.flatten().min().item(), global_step)





                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()