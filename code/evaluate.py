import time
import argparse
import torch
import vmas
import numpy as np
from PIL import Image
import sys

try:
    from train_ippo import Agent as IPPOAgent
    from train_mappo import MAPPOAgent as MAPPOBaseAgent
    from train_mappo_improved import MAPPOAgent as MAPPOImpAgent
    from train_cppo import CPPOAgent
except ImportError:
    pass


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="mappo_improved",
                        choices=["ippo", "mappo", "mappo_improved", "cppo"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 正在评估算法: {args.algo} ===")

    env = vmas.make_env(
        scenario="transport",
        num_envs=1,
        device=device,
        continuous_actions=True,
        max_steps=1000,
        seed=101
    )

    # === 2. 加载模型 ===
    obs = env.reset()
    n_agents = len(obs)
    obs_dim = obs[0].shape[1]
    if hasattr(env.action_space, "spaces"):
        act_dim = env.action_space[0].shape[0]
    else:
        act_dim = env.action_space.shape[0]

    path_map = {
        "ippo": ("ippo_transport_params.pth", IPPOAgent, "independent"),
        "mappo": ("mappo_transport_baseline.pth", MAPPOBaseAgent, "centralized"),
        "mappo_improved": ("mappo_transport_improved.pth", MAPPOImpAgent, "centralized"),
        "cppo": ("cppo_transport_params.pth", CPPOAgent, "cppo")
    }

    model_path, AgentClass, model_type = path_map[args.algo]

    try:
        if args.algo == "ippo":
            agent = AgentClass(obs_dim, act_dim).to(device)
        else:
            agent = AgentClass(obs_dim, act_dim, n_agents).to(device)

        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except:
            state_dict = torch.load(model_path, map_location=device)

        agent.load_state_dict(state_dict)
        agent.eval()
        print(f"模型加载成功: {model_path}")
    except FileNotFoundError:
        print(f"找不到模型文件 {model_path}")
        return

    # === 3. 开始模拟 ===
    frames = []
    total_reward = 0
    next_obs = torch.stack(obs, dim=1)

    with torch.no_grad():
        for step in range(200):
            try:
                frame_data = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=True
                )

                if frame_data is not None:
                    if isinstance(frame_data, torch.Tensor):
                        frame_data = frame_data.cpu().numpy()

                    if isinstance(frame_data, np.ndarray) and frame_data.size > 0:
                        # 确保是 uint8 0-255
                        if frame_data.dtype != np.uint8 and np.max(frame_data) <= 1.0:
                            frame_data = (frame_data * 255).astype(np.uint8)
                        elif frame_data.dtype != np.uint8:
                            frame_data = frame_data.astype(np.uint8)

                        frames.append(Image.fromarray(frame_data))
                else:
                    if step == 0: print("env.render() 返回了 None")

            except Exception as e:
                if step == 0:
                    print(f"渲染功能异常 : {e}")

            # === 智能体动作推理 ===
            if model_type == "independent":
                action, _, _, _ = agent.get_action_and_value(next_obs.view(-1, obs_dim))
                reshaped_action = action.view(1, n_agents, act_dim)
            elif model_type == "centralized":
                action = agent.actor_mean(next_obs.view(-1, obs_dim))
                reshaped_action = action.view(1, n_agents, act_dim)
            elif model_type == "cppo":
                action, _, _, _ = agent.get_action_and_value(next_obs.view(1, -1))
                reshaped_action = action.view(1, n_agents, act_dim)

            action_list = [reshaped_action[:, i, :].clamp(-1.0, 1.0) for i in range(n_agents)]
            next_obs_list, rewards, done, _ = env.step(action_list)

            rew_val = sum(rewards).item()
            total_reward += rew_val
            next_obs = torch.stack(next_obs_list, dim=1)

            if step % 50 == 0:
                print(f"Step {step}: Current Step Reward = {rew_val:.6f} (Total: {total_reward:.4f})")

            if done:
                break

    print(f" 总奖励: {total_reward:.4f}")

    # === 4. 保存 GIF  ===
    if len(frames) > 0:
        save_path = f"simulation_{args.algo}.gif"
        print(f" 正在保存 GIF 到: {save_path}")
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0
        )
    else:
        print("1")


if __name__ == "__main__":
    evaluate()
