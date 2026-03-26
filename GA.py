import numpy as np
from env.MyEnvPara_HPSAC import MyEnv
import argparse
from utils.configs_tools import get_defaults_yaml_args
from pathlib import Path
import swanlab
import yaml


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

def Initialize(N, N_chrom, chrom_range):
    # 初始化染色体矩阵
    # 生成一个 N x N_chrom 的随机矩阵
    chrom_new = np.random.rand(N, N_chrom)

    # 遍历每一列，调整值的范围
    for i in range(N_chrom):
        high = chrom_range.high[i]
        low = chrom_range.low[i]
        chrom_new[:, i] = chrom_new[:, i] * (high- low) + low
    return chrom_new


def CalFitness(chrom, N, env, step, actions):
    fitness = np.zeros((N,), dtype=float)
    CRB = np.zeros((N,), dtype=float)
    comm_rate = np.zeros((N,), dtype=float)
    for i in range(N):
        s = env.reset()
        for nnn in range(step):
            (
                obs,
                rewards,
                dones,
                available_actions,
            ) = env.step(actions[nnn])

        action = chrom[i, :]
        (
            obs,
            r,
            _,
            crb,
        ) = env.step(action)
        fitness[i] = r[0][0]
        CRB[i] = crb[0][0]
        comm_rate[i] = env.comm_uav_data
    return fitness, CRB, comm_rate


def FindBest(chrom, fitness, N_chrom):
    # 初始化chrom_best数组，大小为 (1, N_chrom+1)，最后一个位置存储适应度
    chrom_best = np.zeros(N_chrom + 1)

    # 找到fitness中的最大值及其索引
    maxCorr = np.argmax(fitness)  # 获取最大适应度的索引
    maxNum = fitness[maxCorr]  # 获取最大适应度值

    # 将最优染色体复制到chrom_best中
    chrom_best[:N_chrom] = chrom[maxCorr, :]
    chrom_best[-1] = maxNum  # 最后一位是适应度值

    return chrom_best, maxCorr


def CalAveFitness(fitness):
    # 计算适应度的平均值
    N = fitness.shape[0]  # 获取适应度数组的行数
    fitness_ave = np.sum(fitness) / N  # 计算适应度的平均值
    return fitness_ave


def IfOut(value, value_range, j):
    # 检查并修正超出范围的值
    high = chrom_range.high[j]
    low = chrom_range.low[j]
    if value < low:
        return low
    elif value > high:
        return high
    else:
        return value


def MutChrom(chrom, mut, N, N_chrom, chrom_range, t, iter):
    for i in range(N):
        for j in range(N_chrom):
            mut_rand = np.random.rand()  # 随机生成变异概率
            if mut_rand <= mut:  # 是否进行变异
                mut_pm = np.random.rand()  # 随机生成增加或减少的概率
                mut_num = np.random.rand() * (1 - t / iter) ** 2  # 变异强度

                if mut_pm <= 0.5:  # 增加或减少
                    chrom[i, j] = chrom[i, j] * (1 - mut_num)
                else:
                    chrom[i, j] = chrom[i, j] * (1 + mut_num)


                # 确保染色体值在范围内
                chrom[i, j] = IfOut(chrom[i, j], chrom_range, j)

    chrom_new = chrom
    return chrom_new


def AcrChrom(chrom, acr, N, N_chrom):
    for i in range(N):
        acr_rand = np.random.rand()  # 随机生成交叉概率
        if acr_rand < acr:  # 如果进行交叉操作
            # acr_chrom = np.random.randint(0, N)  # 选择要交叉的染色体
            # acr_node = np.random.randint(0, N_chrom)  # 选择要交叉的节点
            #
            # # 交叉操作，从 `acr_node` 到 `N_chrom` 进行交换
            # chrom[i, acr_node:] = chrom[acr_chrom, acr_node:].copy()
            # chrom[acr_chrom, acr_node:] = chrom[i, acr_node:].copy()
            j = np.random.randint(0, N)        # mate
            k = np.random.randint(0, N_chrom)  # cut point
            tmp = chrom[i, k:].copy()
            chrom[i, k:], chrom[j, k:] = chrom[j, k:], tmp

    chrom_new = chrom
    return chrom_new


def ReplaceWorse(chrom, chrom_best, fitness):
    max_num = np.max(fitness)  # 最大适应度值
    min_num = np.min(fitness)  # 最小适应度值
    limit = (max_num - min_num) * 0.2 + min_num  # 计算适应度的阈值

    # 找到适应度低于阈值的染色体
    replace_corr = fitness < limit

    # 计算要替换的染色体数量
    replace_num = np.sum(replace_corr)

    # 将适应度低的染色体替换为最优染色体
    chrom[replace_corr, :] = np.tile(chrom_best[:-1], (replace_num, 1))  # 替换染色体
    fitness[replace_corr] = np.tile(chrom_best[-1], replace_num)  # 替换适应度

    chrom_new = chrom
    fitness_new = fitness

    return chrom_new, fitness_new


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

    # swanlab.init(
    #     project="GA",
    #     name="GA",
    #     # entity="kangyan-uestc",
    #     config={}
    # )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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

    if env_args["algo"] == "hsac":
        select_model = 0
    elif env_args["algo"] == "pdqn":
        select_model = 1
    elif env_args["algo"] == "sac":
        select_model = 2

    env = MyEnv(env_args)
    agent_num = 1
    gene_len = np.zeros(agent_num, dtype=np.int64)
    action_total = 0
    for i in range(agent_num):
        gene_len[i] = env.action_space.shape[0]
        action_total += env.action_space.shape[0]
    gene_len_total = np.zeros(action_total, dtype=np.float64)

    # start_time = time.time()
    # 基础参数
    N = 50  # 种群内个体数目
    N_chrom = action_total  # 一条染色体上的基因数
    iter = 100  # 迭代次数
    mut = 0.02  # 突变概率
    acr = 0.8  # 交叉概率
    # best = 1

    chrom_range = env.action_space  # 每个节点的值的区间

    # chrom = np.zeros((N, N_chrom))  # 存放染色体的矩阵
    fitness = np.zeros(N)  # 存放染色体的适应度
    fitness_ave = np.zeros(iter)  # 存放每一代的平均适应度
    fitness_best = np.zeros(iter)  # 存放每一代的最优适应度
    chrom_best = np.zeros((N_chrom,), dtype=np.float64)  # 存放当前代的最优染色体与适应度
    best_actions = []
    # 初始化
    chrom = Initialize(N, N_chrom, chrom_range)
    # rows = np.split(chrom, [gene_len[0], gene_len[0] + gene_len[1]])
    actions = [[] for _ in range(env.total_step)]
    for step in range(env.total_step):
        total_reward = 0
        evaluate_pcrb = []

        fitness, CRB, comm_rate = CalFitness(chrom, N, env, step, best_actions)
        chrom_best, index_max = FindBest(chrom, fitness, N_chrom)
        fitness_best[0] = chrom_best[-1]
        fitness_ave[0] = CalAveFitness(fitness)

        for t in range(1, iter):
            chrom = MutChrom(chrom, mut, N, N_chrom, chrom_range, t, iter)
            chrom = AcrChrom(chrom, acr, N, N_chrom)
            fitness, CRB, comm_rate = CalFitness(chrom, N, env, step, best_actions)
            chrom_best_temp, index_max = FindBest(chrom, fitness, N_chrom)
            if chrom_best_temp[-1] > chrom_best[-1]:
                chrom_best = chrom_best_temp
            chrom, fitness = ReplaceWorse(chrom, chrom_best, fitness)
            fitness_best[t] = chrom_best[-1]
            fitness_ave[t] = CalAveFitness(fitness)
        action = chrom_best
        best_actions.append(action)
        print(f"step：{step}, frame: {action[0]:.6f}, comm_rate: {comm_rate[index_max]:.6f}, reward: {action[-1]:.6f}, CRB: {CRB[index_max]:.6f} \r\n")


    s = env.reset()
    rewards = 0
    average_crb = 0
    for step in range(env.total_step):
        (
            obs,
            r,
            dones,
            infos,
        ) = env.step(best_actions[step])
        rewards += r[0][0]
        average_crb = infos[0][0]
    # swanlab.log({"PCRB": infos[0]["sensing_pcrb"]})
    # swanlab.log({"rewards": rewards})
    print(f"rewards：{rewards:.6f}")
    print(f"CRB：{average_crb:.6f}")
    swanlab.log({"evaluate_reward": rewards})
    swanlab.log({"evaluate_average_crb": average_crb})
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"代码运行时间：{execution_time:.6f} 秒")
