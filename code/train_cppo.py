import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import vmas


# ==========================================
# 1. CPPO 神经网络定义
# ==========================================
class CPPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, n_agents, hidden_dim=128):
        super(CPPOAgent, self).__init__()
        self.n_agents = n_agents
        self.state_dim = obs_dim * n_agents
        self.joint_act_dim = act_dim * n_agents

        # Centralized Critic
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Centralized Actor
        self.actor_mean = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, self.joint_act_dim), nn.Tanh()
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.joint_act_dim))

    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, state, action=None):
        action_mean = self.actor_mean(state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        probs = Normal(action_mean, torch.exp(action_logstd))
        if action is None: action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(state)


# ==========================================
# 2. 训练主循环
# ==========================================
def train():
    parser = argparse.ArgumentParser(description="VMAS CPPO Implementation")
    parser.add_argument("--scenario", type=str, default="transport")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--total-timesteps", type=int, default=5000000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"CPPO Running on device: {device}")

    env = vmas.make_env(
        scenario=args.scenario, num_envs=args.num_envs, device=device,
        continuous_actions=True, max_steps=args.max_steps, seed=42
    )

    obs = env.reset()
    n_agents = len(obs)
    obs_dim = obs[0].shape[1]
    if hasattr(env.action_space, "spaces"):
        act_dim = env.action_space[0].shape[0]
    else:
        act_dim = env.action_space.shape[0]

    agent = CPPOAgent(obs_dim, act_dim, n_agents).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # Buffers
    state_buf = torch.zeros((args.num_steps, args.num_envs, agent.state_dim)).to(device)
    act_buf = torch.zeros((args.num_steps, args.num_envs, agent.joint_act_dim)).to(device)
    logp_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rew_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    don_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    val_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs = torch.stack(env.reset(), dim=1)
    next_state = next_obs.view(args.num_envs, -1)
    next_done = torch.zeros(args.num_envs).to(device)

    # 用于存储曲线数据
    all_episode_rewards = []

    num_updates = args.total_timesteps // (args.num_steps * args.num_envs)

    for update in range(1, num_updates + 1):
        agent.eval()
        for step in range(args.num_steps):
            global_step += args.num_envs
            state_buf[step], don_buf[step] = next_state, next_done

            with torch.no_grad():
                act, logp, _, val = agent.get_action_and_value(next_state)
                val_buf[step], act_buf[step], logp_buf[step] = val.view(-1), act, logp

            reshaped_act = act.view(args.num_envs, n_agents, act_dim)
            act_list = [reshaped_act[:, i, :].clamp(-1.0, 1.0) for i in range(n_agents)]
            next_obs_list, rewards_list, dones, _ = env.step(act_list)

            next_obs = torch.stack(next_obs_list, dim=1)
            next_state = next_obs.view(args.num_envs, -1)
            next_done = dones.float()
            rew_buf[step] = torch.stack(rewards_list, dim=1).sum(dim=1)  # Team Reward

        # GAE
        with torch.no_grad():
            next_val = agent.get_value(next_state).view(-1)
            adv = torch.zeros_like(rew_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                nonterminal = 1.0 - next_done if t == args.num_steps - 1 else 1.0 - don_buf[t + 1]
                nextvalues = next_val if t == args.num_steps - 1 else val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nextvalues * nonterminal - val_buf[t]
                adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nonterminal * lastgaelam
            ret = adv + val_buf

        # Train
        agent.train()
        b_st, b_act, b_logp = state_buf.reshape(-1, agent.state_dim), act_buf.reshape(-1,
                                                                                      agent.joint_act_dim), logp_buf.reshape(
            -1)
        b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)

        inds = np.arange(b_st.shape[0])
        for _ in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, b_st.shape[0], b_st.shape[0] // args.num_minibatches):
                mb_inds = inds[start:start + b_st.shape[0] // args.num_minibatches]
                _, newlogp, ent, newval = agent.get_action_and_value(b_st[mb_inds], b_act[mb_inds])

                ratio = (newlogp - b_logp[mb_inds]).exp()
                mb_a = b_adv[mb_inds]
                mb_a = (mb_a - mb_a.mean()) / (mb_a.std() + 1e-8)
                pg_loss = torch.max(-mb_a * ratio,
                                    -mb_a * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
                v_loss = 0.5 * ((newval.view(-1) - b_ret[mb_inds]) ** 2).mean()
                loss = pg_loss - args.ent_coef * ent.mean() + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm);
                optimizer.step()

        if update % 10 == 0:
            avg_rew = rew_buf.mean().item()
            all_episode_rewards.append(avg_rew)
            print(f"Update {update}/{num_updates} | Step {global_step} | Mean Team Reward: {avg_rew:.4f}")

    np.save("rewards_cppo.npy", np.array(all_episode_rewards))
    torch.save(agent.state_dict(), "cppo_transport_params.pth")
    print("CPPO Training finished. Data saved to rewards_cppo.npy")


if __name__ == "__main__":
    train()
