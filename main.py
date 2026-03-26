import torch
import numpy as np
from env.MyEnvPara_HPSAC import MyEnv
from agent.MyPDQN import PDQN
from agent.MyHPSAC import HPSAC
from agent.SAC import SAC
from utils.configs_tools import get_defaults_yaml_args
import argparse
from buffer.replay_buffer import ReplayBuffer
from utils.models_tools import init_device
from utils.normalization_onpolicy import Normalization, RewardScaling
from pathlib import Path
import yaml
import swanlab


def update_cfg(cfg: dict, updates: dict):
    """递归地把 updates 合并进 cfg."""
    for k, v in updates.items():
        if isinstance(v, dict) and k in cfg:
            update_cfg(cfg[k], v)
        else:
            cfg[k] = v


def parse_kv_pairs(pairs):
    """把形如 model.lr=0.01 data.batch_size=128 解析成嵌套字典."""
    upd = {}
    for p in pairs:
        key, val = p.split("=", 1)
        # 尝试把数字/布尔值从字符串变成对应类型
        try:
            val = eval(val)
        except Exception:
            pass
        levels = key.split(".")
        d = upd
        for l in levels[:-1]:
            d = d.setdefault(l, {})
        d[levels[-1]] = val
    return upd

def evaluate_policy(env, agent):
    times = 1  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    evaluate_average_crb = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        while episode_steps < env.total_step:
            episode_steps += 1
            # start_time = time.time()
            a = agent.choose_action(s, deterministic=True)  # We use the deterministic policy during the evaluating
            # end_time = time.time()
            # execution_time = end_time - start_time
            # print(f"代码运行时间：{execution_time:.6f} 秒")
            s_, r, done, average_crb = env.step(a)
            episode_reward += r
            s = s_

        evaluate_reward += episode_reward
        evaluate_average_crb += average_crb
    return int(evaluate_reward / times), float(evaluate_average_crb / times)


def reward_adapter(r, env_index):
    if env_index == 0:  # Pendulum-v1
        r = (r + 8) / 8
    elif env_index == 1:  # BipedalWalker-v3
        if r <= -100:
            r = -1
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--cfg", type=Path, default="./configs/envs_cfgs/IAPSC.yaml",
                        help="YAML 配置文件路径")
    parser.add_argument("override", nargs="*", help="用 key=value 覆盖配置，支持嵌套，如 model.lr=0.01")
    parser.add_argument("--save", action="store_true",
                        help="把修改后的配置写回原文件")
    args = parser.parse_args()
    # 1. 读取 YAML
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. 解析命令行覆盖项
    updates = parse_kv_pairs(args.override)

    # 3. 合并
    update_cfg(cfg, updates)

    # 4. 选写：保存或直接使用
    if args.save:
        with open(args.cfg, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"已保存更新后的配置到 {args.cfg}")

    parser.add_argument(
        "--algo",
        type=str,
        default="pdqn",
        choices=[
            "hsac",
            "pdqn",
            "sac",
            "ga",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="IAPSC",
        choices=[
            "IAPSC",
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
    )

    args, unparsed_args = parser.parse_known_args()
    args = vars(args)  # convert to dict
    algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])

    swanlab.init(
        project="Hybrid-SAC",
        name="lr-{}-algo-{}-power-{}-commdata-{}-frame-{}-uav_assisted-{}-seed-{}.".
        format(
            env_args["learning_rate"],
            env_args["algo"],
            env_args["bs_power"],
            env_args["comm_uav_data_min"]/8000000,
            env_args["T"],
            env_args["uav_assisted"],
            env_args["seed"]),
        # entity="kangyan-uestc",
        config={**algo_args, **env_args}
    )

    device = init_device(algo_args["device"])

    env = MyEnv(env_args)
    env_evaluate = MyEnv(env_args)
    if env_args["algo"] == "hsac":
        select_model = 0
    elif env_args["algo"] == "pdqn":
        select_model = 1
    elif env_args["algo"] == "sac":
        select_model = 2

    print("algo={}".format(env_args["algo"]))
    print("seed={}".format(env_args["seed"]))

    s = env.reset()
    # Set random seed
    seed = env_args["seed"]
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    para_action_dim = env.para_action_dim
    dis_action_dim = env.dis_action_dim

    max_action = float(env.action_space.high[1])
    dis_action_max = env.dis_action_max

    max_episode_steps = env.total_step  # Maximum number of steps per episode
    env.max_episode_steps = max_episode_steps
    max_episode_steps = env.max_episode_steps  # Maximum number of steps per episode

    if select_model == 0:
        agent = HPSAC(state_dim, para_action_dim, dis_action_max, max_action, env_args)
    elif select_model == 1:
        agent = PDQN(state_dim, para_action_dim, dis_action_max, max_action, env_args)
    else:
        agent = SAC(state_dim, action_dim, max_action, env_args)

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    max_train_steps = 200000  # Maximum number of training steps
    random_steps = 50000  # Take the random actions in the beginning for the better exploration
    evaluate_freq = 400  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_freq = env_args["evaluate_freq"]
    evaluate_num = 0  # Record the number of evaluations
    learn_freq = 2
    evaluate_rewards = []  # Record the rewards during the evaluating
    evaluate_average_crbs = []
    actor_losses = []
    critic_losses = []
    total_steps = 0  # Record the total steps during the training
    rewards = []
    episode_num = 0
    # state_norm = Normalization(shape=state_dim)  # Trick 2:state normalization
    # reward_scaling = RewardScaling(shape=1, gamma=0.99)
    while total_steps < max_train_steps:
        s = env.reset()
        # s = state_norm(s)
        # reward_scaling.reset()
        episode_steps = 0
        evaluate_reward = 0
        done = False
        episode_num += 1
        while episode_steps < max_episode_steps:
            episode_steps += 1
            if replay_buffer.size < random_steps:  # Take the random actions in the beginning for the better exploration
                a = env.action_space.sample()
                a_ = a
                a_[0] = int(a_[0].clip(1, dis_action_dim))
            else:
                total_steps += 1
                if select_model == 2:
                    a = agent.choose_action(s)
                    a_ = a
                    a_[0] = int(((a_[0] + 1) /2 * env.action_space.high[0]).clip(1, dis_action_dim))
                else:
                    a = agent.choose_action(s)
                    a_ = a
            s_, r_, done_, _ = env.step(a_)
            # s_ = state_norm(s_)
            # r_ = reward_scaling(r_)

            # r = reward_adapter(r, env_index)  # Adjust rewards for better performance
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != max_episode_steps:
                dw = True
            else:
                dw = False
            replay_buffer.store(s, a, r_, s_, dw)  # Store the transition
            s = s_
            evaluate_reward += r_

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0 and replay_buffer.size >= random_steps:

                if select_model == 1:
                    agent.VAR *= .999  # decay the action randomness, for a smaller var of gaussian value
                    agent.GATE *= .999
                    print("VAR:{}".format(agent.VAR))
                    print("GATE:{}".format(agent.GATE))
                evaluate_num += 1
                evaluate_reward, evaluate_average_crb = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                evaluate_average_crbs.append(evaluate_average_crb)
                print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))
                print("evaluate_num:{} \t evaluate_average_crb:{}".format(evaluate_num, evaluate_average_crb))
                swanlab.log({"evaluate_reward": evaluate_reward})
                swanlab.log({"evaluate_average_crb": evaluate_average_crb})
                a = 1

            if (total_steps + 1) % learn_freq == 0 and replay_buffer.size >= random_steps:
                actor_loss, critic_loss = agent.learn(replay_buffer)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                # evaluate_reward = evaluate_policy(env_evaluate, agent)
                # evaluate_rewards.append(evaluate_reward)
                # print("num:{} \t reward:{}".format(total_steps, evaluate_reward))


        # evaluate_num += 1
        # evaluate_rewards.append(evaluate_reward)
        # print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))

    # TODO Save Network Paras
    # torch.save(agent.actor_1.state_dict(), 'dqn_maze_model.pkl')
    # self.actor_1.load_state_dict(torch.load('dqn_maze_model.pkl'))  # 加载模型
    # if select_model == 0:
    #     np.save('./data_train/HPSAC_crb_{}.npy'.format(seed),
    #             np.array(evaluate_average_crbs))
    #     np.save('./data_train/HPSAC_reward_{}.npy'.format(seed),
    #             np.array(evaluate_rewards))
    #     np.save('./data_train/HPSAC_actor_loss_{}.npy'.format(seed),
    #             np.array(actor_losses))
    #     np.save('./data_train/HPSAC_critic_loss_{}.npy'.format(seed),
    #             np.array(critic_losses))
    #     torch.save(agent.actor.state_dict(), 'data/HPSAC_model_{}.pkl'.format(seed))
    # elif select_model == 1:
    #     np.save('./data_train/PDQN_reward_{}.npy'.format(seed),
    #             np.array(evaluate_rewards))
    #     np.save('./data_train/PDQN_actor_loss_{}.npy'.format(seed), np.array(actor_losses))
    #     np.save('./data_train/PDQN_critic_loss_{}.npy'.format(seed), np.array(actor_losses))
    #     torch.save(agent.actor.state_dict(), 'data/PDQN_model_{}.pkl'.format(seed))
    # else:
    #     np.save('./data_train/SAC_reward_{}.npy'.format(seed),
    #             np.array(evaluate_rewards))
    #     np.save('./data_train/SAC_actor_loss_{}.npy'.format(seed), np.array(actor_losses))
    #     np.save('./data_train/SAC_critic_loss_{}.npy'.format(seed), np.array(actor_losses))
    replay_buffer = None
