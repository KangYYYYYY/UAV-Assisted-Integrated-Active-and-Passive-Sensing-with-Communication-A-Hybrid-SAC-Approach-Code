from copy import deepcopy

import gym
from gym import spaces
import time
import scipy.io as scio
import numpy as np
from gym.utils import seeding
import cmath


# running
# ./pso > pso.file 2>&1 &
def angles_from_positions(origin_pos, look_pos):
    """Return (θ, φ) degrees from *origin* pointing toward *look_pos*.

    θ: elevation (0 ° = broadside +z, 90 ° = xy‑plane).
    φ: azimuth   (0 ° = +x, CCW toward +y).
    """
    o = np.asarray(origin_pos, dtype=np.float32)
    t = np.asarray(look_pos, dtype=np.float32)
    v = t - o
    r = np.linalg.norm(v)
    if r == 0:
        raise ValueError("Origin and look positions coincide — undefined angles.")
    dx, dy, dz = v
    phi = np.arctan2(dy, dx)  # [-π, π]
    theta = np.arccos(dz / r)  # [0, π]
    return np.degrees(theta), np.degrees(phi)


def upa_steering_vector(theta_deg, phi_deg,
                        num_rows=4, num_cols=4, d=0.5):
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    m = np.arange(num_cols)  # x 方向索引
    n = np.arange(num_rows)[:, None]  # y 方向索引
    phase = 2 * np.pi * d * (m * np.sin(theta) * np.cos(phi) +
                             n * np.sin(theta) * np.sin(phi))
    vec = np.exp(1j * phase).ravel()
    return vec / np.sqrt(num_rows * num_cols)


def upa_tx_vector_from_positions(tx_pos, rx_pos, **upa_kw):
    theta, phi = angles_from_positions(tx_pos, rx_pos)
    return upa_steering_vector(theta, phi, **upa_kw)


def upa_rx_vector_from_positions(rx_pos, tx_pos, **upa_kw):
    theta, phi = angles_from_positions(rx_pos, tx_pos)
    return upa_steering_vector(theta, phi, **upa_kw)


def upa_tx_vector_to_sensing(tx_pos, sensing_pos, **upa_kw):
    """Steering vector at TX pointing to *sensing_pos*."""
    theta, phi = angles_from_positions(tx_pos, sensing_pos)
    return upa_steering_vector(theta, phi, **upa_kw)


def upa_rx_vector_from_sensing(rx_pos, sensing_pos, **upa_kw):
    """Response vector at RX for signal arriving from *sensing_pos*."""
    theta, phi = angles_from_positions(rx_pos, sensing_pos)
    return upa_steering_vector(theta, phi, **upa_kw)


def comm_channel_matrix(tx_pos, rx_pos,
                        path_gain_ref: np.float32 = 1.0,
                        num_tx_rows: int = 4, num_tx_cols: int = 4,
                        num_rx_rows: int = 4, num_rx_cols: int = 4,
                        d_tx: np.float32 = 0.5, d_rx: np.float32 = 0.5):
    a_t = upa_tx_vector_from_positions(tx_pos, rx_pos,
                                       num_rows=num_tx_rows, num_cols=num_tx_cols, d=d_tx)
    a_r = upa_rx_vector_from_positions(rx_pos, tx_pos,
                                       num_rows=num_rx_rows, num_cols=num_rx_cols, d=d_rx)

    a = np.linalg.norm(rx_pos - tx_pos)
    gain = np.sqrt(path_gain_ref) / np.linalg.norm(rx_pos - tx_pos)
    return gain * np.outer(a_r, a_t.conj())


def sensing_channel_matrix(tx_pos, rx_pos, sensing_pos,
                           rcs: np.float32 = 0.01,
                           path_gain_ref: np.float32 = 1.0,
                           num_tx_rows: int = 4, num_tx_cols: int = 4,
                           num_rx_rows: int = 4, num_rx_cols: int = 4,
                           d_tx: np.float32 = 0.5, d_rx: np.float32 = 0.5):
    a_t = upa_tx_vector_to_sensing(tx_pos, sensing_pos,
                                   num_rows=num_tx_rows, num_cols=num_tx_cols, d=d_tx)
    a_r = upa_rx_vector_from_sensing(rx_pos, sensing_pos,
                                     num_rows=num_rx_rows, num_cols=num_rx_cols, d=d_rx)

    gain = np.sqrt(path_gain_ref) / np.linalg.norm(rx_pos - sensing_pos) / np.linalg.norm(tx_pos - sensing_pos)
    return rcs * gain * np.outer(a_r, a_t.conj())


class MyEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, env_args=None):
        self.rician_factor = env_args["rician_factor"]
        self.uav_assisted = env_args["uav_assisted"]
        self.GA_select = 0
        self.sens_uav_bs_y_theta = None
        self.sens_uav_bs_x_theta = None
        self.bs_sens_uav_y_theta = None
        self.array_vector_bs_sens_uav_y = None
        self.array_vector_bs_sens_uav_x = None
        self.bs_sens_uav_x_theta = None
        self.sens_comm_uav_h_dim = None
        self.state_sens_comm_uav_h = None
        self.select_state_model = 0
        self.bs_comm_uav_h_dim = None
        self.bs_sens_uav_h_dim = None
        self.state_bs_comm_uav_h = None
        self.state_bs_sens_uav_h = None
        self.np_random = None

        self.counts = 0
        self.channel_comm_gain_1m = env_args["channel_comm_gain_1m"]
        self.max_episode_steps = env_args["max_episode_steps"]

        self.channel_comm_gain_1m = env_args["channel_comm_gain_1m"]
        self.channel_comm_gain_1m = np.power(10, self.channel_comm_gain_1m / 10)
        self.channel_sens_gain_1m = env_args["channel_sens_gain_1m"]
        self.channel_sens_gain_1m = np.power(10, self.channel_sens_gain_1m / 10)
        self.band_width = env_args["band_width"]
        self.bs_power = env_args["bs_power"]
        self.T = env_args["T"]
        self.time = env_args["time"]

        self.bs_tx_upa_x_num = env_args["bs_tx_upa_x_num"]
        self.bs_tx_upa_y_num = env_args["bs_tx_upa_y_num"]
        self.bs_tx_upa_num = self.bs_tx_upa_x_num * self.bs_tx_upa_y_num
        self.bs_rx_upa_x_num = env_args["bs_rx_upa_x_num"]
        self.bs_rx_upa_y_num = env_args["bs_rx_upa_y_num"]
        self.bs_rx_upa_num = self.bs_rx_upa_x_num * self.bs_rx_upa_y_num
        self.comm_uav_rx_upa_x_num = env_args["comm_uav_rx_upa_x_num"]
        self.comm_uav_rx_upa_y_num = env_args["comm_uav_rx_upa_y_num"]
        self.comm_uav_rx_upa_num = self.comm_uav_rx_upa_x_num * self.comm_uav_rx_upa_y_num

        self.noise_power = env_args["noise_power"]
        self.noise_power = np.power(10, self.noise_power / 10) * 0.001

        # init sensing uav rcs
        self.sens_uav_rcs = env_args["sens_uav_rcs"]

        # init bs and uav location and distance
        self.bs_location = np.array([0, 0, 0])
        self.unknown_uav_num = 1
        self.unknown_uav_location = np.array([300, 300, 100])
        self.unknown_uav_location_init = np.array([300, 300, 100])
        self.unknown_uav_location_end = np.array([300, 300, 100])

        self.known_uav_num = 1
        self.known_uav_location = np.array([-0, 200, 100])
        self.known_uav_location_init = np.array([0, 200, 100])
        self.known_uav_location_end = np.array([200, 200, 100])

        self.total_step = env_args["max_episode_steps"]
        self.now_step = 0
        self.known_uav_location_buffer = np.zeros((self.total_step, 3), dtype=float)
        self.unknown_uav_location_buffer = np.zeros((self.total_step, 3), dtype=float)

        self.bs_isac_num = self.unknown_uav_num + self.known_uav_num
        self.bs_isac_beam_num = self.unknown_uav_num * 2 + self.known_uav_num
        self.crb_max = env_args["crb_max"]
        self.comm_uav_data_min = env_args["comm_uav_data_min"]

        self.punish_factor = env_args["punish_factor"]

        self.bs_antenna_distance = 1
        self.ba_antenna_wavelength = 2
        self.distance_bs_uav = np.zeros((self.total_step, 2), dtype=float)
        self.distance_uav_uav = np.zeros((self.total_step,), dtype=float)

        # init ula num of bs and comm uav

        # init RL action
        # including: bs beta, power, and beamforming matrix
        self.beta_action = 0.5 * self.T
        self.beta_action_init = 0.5 * self.T

        self.power_sens_action = 0.5
        self.power_sens_action_init = 0.5

        # pure sensing , isac , comm
        self.isac_beam_vector = np.zeros((self.bs_tx_upa_num, self.bs_isac_num), dtype=np.complex128)
        self.pure_sens_beam_action = np.zeros((1, self.bs_tx_upa_num * 2), dtype=np.complex128)
        self.pure_sens_beam_vector = np.zeros((self.bs_tx_upa_num, 1), dtype=np.complex128)
        self.isac_beam_action = np.zeros((self.bs_tx_upa_num, self.bs_isac_num), dtype=complex)
        self.bs_beam_matrix_action = np.zeros((self.bs_tx_upa_num, self.bs_isac_num), dtype=complex)
        self.bs_beam_matrix_action[:, :] = 0.7 + 0.7j

        self.bs_beam_matrix_action_init = np.zeros((self.bs_tx_upa_num * 2, self.bs_isac_beam_num),
                                                   dtype=float)
        self.bs_beam_matrix_action_init[:, :] = 0.25

        self.comm_h = np.zeros((self.total_step, self.comm_uav_rx_upa_num, self.bs_tx_upa_num),
                               dtype=np.complex128)
        self.active_sens_h = np.zeros((self.total_step, self.bs_rx_upa_num, self.bs_tx_upa_num),
                                      dtype=np.complex128)
        self.passive_sens_h = np.zeros((self.total_step, self.comm_uav_rx_upa_num, self.bs_tx_upa_num),
                                       dtype=np.complex128)

        self.g_xx_bs = np.zeros((self.total_step,), dtype=np.float64)
        self.g_xy_bs = np.zeros((self.total_step,), dtype=np.float64)
        self.g_xz_bs = np.zeros((self.total_step,), dtype=np.float64)
        self.g_yy_bs = np.zeros((self.total_step,), dtype=np.float64)
        self.g_yz_bs = np.zeros((self.total_step,), dtype=np.float64)
        self.g_zz_bs = np.zeros((self.total_step,), dtype=np.float64)

        self.g_xx_comm = np.zeros((self.total_step,), dtype=np.float64)
        self.g_xy_comm = np.zeros((self.total_step,), dtype=np.float64)
        self.g_xz_comm = np.zeros((self.total_step,), dtype=np.float64)
        self.g_yy_comm = np.zeros((self.total_step,), dtype=np.float64)
        self.g_yz_comm = np.zeros((self.total_step,), dtype=np.float64)
        self.g_zz_comm = np.zeros((self.total_step,), dtype=np.float64)

        self.init_sensing_vector()

        self.sens_para = 8 * np.pi * np.pi * self.band_width / self.noise_power / (3 * np.power(10, 8))

        # init reward parameters
        self.reward = 0
        self.crb = 0
        self.crb_total = 0
        self.crb_total_GA = 0
        self.crb_average = 0
        self.crb_average_GA = 0

        # init action and observation
        self.para_action_dim = self.bs_isac_beam_num * self.bs_tx_upa_num * 2
        self.dis_action_dim = 1
        self.dis_action_max = self.T

        self.action_dim = self.para_action_dim + self.dis_action_dim
        self.action = np.zeros((self.action_dim,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.action_space.high[0] = self.T - 1
        self.action_space.low[0] = 1

        self.observation_dim = 1 + 1 + 3 + 3
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.observation_dim,), dtype=np.float32)

        self.comm_uav_data = None
        self.comm_uav_rate = None
        self.comm_uav_sinr = None
        self.state = None  # 当前状态

        self.state_1 = None  # 当前状态
        self.state_2 = None  # 当前状态

    def reset(self, start_state=None):
        self.known_uav_location = self.known_uav_location_init
        self.unknown_uav_location = self.unknown_uav_location_init
        self.now_step = 0
        if start_state is None:
            a = np.array([self.beta_action_init]).reshape(1, )
            a = np.concatenate((a, self.bs_beam_matrix_action_init.flatten()), axis=0)
            s, r, done, _ = self.step(a)
            self.state = s
        else:
            if self.observation_space.contains(start_state):
                self.state = start_state
            else:
                self.state = self.observation_space.sample()
        # self.state = np.ones((33,), dtype=float) * 0.5
        self.counts = 0
        self.crb_total = 0
        self.crb = 0
        self.now_step = 0
        return self.state

    def step(self, action):
        # get action
        self.action = np.asarray(action, dtype=np.float64).copy()
        self.action[0] = self.action[0].clip(1, self.dis_action_max)

        # frame length of only sensing part
        # if self.action[0] < 1:
        #     self.beta_action = int(np.clip(1 + np.rint(self.action[0] * (self.T - 2)), 1, self.T - 1))
        # sensing power for ISAC part
        # self.power_sens_action = 0.05 * self.bs_power + self.action[1] * (0.90 * self.bs_power)  # [0.05, 0.95]*P
        # beam for ISAC part
        self.beta_action = self.action[0]
        isac_beam_action = self.action[1: 1 + 2 * self.bs_isac_num * self.bs_tx_upa_num]
        pure_sens_beam_action = self.action[
                                1 + 2 * self.bs_isac_num * self.bs_tx_upa_num:1 + 2 * (
                                            self.bs_isac_num + 1) * self.bs_tx_upa_num]
        isac_ba = np.asarray(isac_beam_action, dtype=float).reshape(self.bs_isac_num, self.bs_tx_upa_num, 2)
        pure_sens_ba = np.asarray(pure_sens_beam_action, dtype=float).reshape(1, self.bs_tx_upa_num, 2)

        self.isac_beam_action = np.zeros((self.bs_isac_num, self.bs_tx_upa_num * 2), dtype=np.complex128)
        self.isac_beam_vector = np.zeros((self.bs_tx_upa_num, self.bs_isac_num), dtype=np.complex128)

        self.pure_sens_beam_action = np.zeros((1, self.bs_tx_upa_num * 2), dtype=np.complex128)
        self.pure_sens_beam_vector = np.zeros((self.bs_tx_upa_num, 1), dtype=np.complex128)

        # 如果还想保留 self.isac_beam_action 的二维实数存储（和原来一致）
        self.isac_beam_action[:] = isac_ba.reshape(self.bs_isac_num, -1)
        self.pure_sens_beam_action[:] = pure_sens_ba.reshape(1, -1)

        # 直接合成复数：(bs_isac_num, bs_tx_upa_num)
        vec = isac_ba[..., 0] + 1j * isac_ba[..., 1]
        # (bs_tx_upa_num, bs_isac_num)
        self.isac_beam_vector[:] = vec.T
        # 功率归一化
        p = np.linalg.norm(self.isac_beam_vector) ** 2
        if p > 0:
            self.isac_beam_vector *= np.sqrt(self.bs_power / p)  # 原地乘，不重新分配

        # 直接合成复数：(bs_isac_num, bs_tx_upa_num)
        vec = pure_sens_ba[..., 0] + 1j * pure_sens_ba[..., 1]
        # (bs_tx_upa_num, bs_isac_num)
        self.pure_sens_beam_vector[:] = vec.T
        # 功率归一化
        p = np.linalg.norm(self.pure_sens_beam_vector) ** 2
        if p > 0:
            self.pure_sens_beam_vector *= np.sqrt(self.bs_power / p)  # 原地乘，不重新分配

        # section -- bs active sense the sensing UAV and communicate with the communication UAV
        self.communication_performance()
        self.crb = self.sensing_performance()

        # get reward
        self.get_reward()

        # normalize sens state
        sens_uav_state = np.zeros((1,), dtype=float)
        sens_uav_state[0] = np.log(self.crb_average / self.crb_max)

        # if sens_uav_state[0] > 1:
        #     sens_uav_state[0] = 1
        self.state = sens_uav_state

        # normalize comm state
        comm_uav_state = np.zeros((1,), dtype=float)
        comm_uav_state[0] = self.comm_uav_data / self.comm_uav_data_min
        # if comm_uav_state[0] > 1:
        #     comm_uav_state[0] = 1
        self.state = np.concatenate((self.state, comm_uav_state), axis=0)

        self.state = np.concatenate((self.state, self.known_uav_location_buffer[self.now_step].reshape(3, )), axis=0)
        self.state = np.concatenate((self.state, self.unknown_uav_location_buffer[self.now_step].reshape(3, )), axis=0)

        self.counts += 1

        self.now_step += 1
        if self.now_step == self.max_episode_steps:
            done = True
        else:
            done = False

        return self.state, self.reward, done, self.crb_average

    def get_reward(self):
        self.crb_total += self.crb
        self.crb_average = self.crb_total / (self.now_step + 1)

        # sens_punish = (self.crb_average - self.crb_max) / self.crb_max
        # if sens_punish > 0:
        #     sens_punish = self.crb_average / self.crb_max
        #     sens_factor = self.punish_factor[0]
        # else:
        #     sens_factor = self.reward_factor[0]
        #     sens_punish = -np.log(self.crb_average / self.crb_max)

        sens_punish = np.log(self.crb_average / self.crb_max)
        sens_factor = self.punish_factor[0]

        comm_punish = (self.comm_uav_data - self.comm_uav_data_min) / self.comm_uav_data_min
        if comm_punish > 0:
            comm_factor = 0
        else:
            comm_factor = self.punish_factor[1]
            comm_punish = -comm_punish

        self.reward = sens_factor * sens_punish + comm_factor * comm_punish

        return

    def sensing_performance(self):

        # sensing beam
        w_sens = self.isac_beam_vector[:, 0].T.reshape(self.bs_tx_upa_num, 1)
        # 功率归一化
        w_sens_power = np.vdot(w_sens, w_sens).real
        if w_sens_power > 0:
            w_sens /= np.sqrt(w_sens_power)  # 原地乘，不重新分配

        # pure sensing beam
        w_sens_pure = self.pure_sens_beam_vector[:, 0].T.reshape(self.bs_tx_upa_num, 1)
        # 计算功率 (向量的二范数平方)
        w_sens_pure_power = np.vdot(w_sens_pure, w_sens_pure).real
        # 归一化，使得总功率 = 1
        w_sens_pure /= np.sqrt(w_sens_pure_power)

        pure_sens_time = int(self.beta_action)

        power = np.zeros((self.T, 1), dtype=float)
        power[0:pure_sens_time, 0] = w_sens_pure_power
        power[pure_sens_time:, 0] = w_sens_power

        g_xx_pure_bs = 0
        g_xy_pure_bs = 0
        g_xz_pure_bs = 0
        g_yy_pure_bs = 0
        g_yz_pure_bs = 0
        g_zz_pure_bs = 0

        g_xx_pure_comm = 0
        g_xy_pure_comm = 0
        g_xz_pure_comm = 0
        g_yy_pure_comm = 0
        g_yz_pure_comm = 0
        g_zz_pure_comm = 0

        g_xx_isac_bs = 0
        g_xy_isac_bs = 0
        g_xz_isac_bs = 0
        g_yy_isac_bs = 0
        g_yz_isac_bs = 0
        g_zz_isac_bs = 0

        pure_active_sens_power = np.power(abs(self.active_sens_h[self.now_step] @ w_sens_pure), 2)

        g_xx_pure_bs += np.sum(pure_active_sens_power) * self.g_xx_bs[self.now_step]
        g_xy_pure_bs += np.sum(pure_active_sens_power) * self.g_xy_bs[self.now_step]
        g_xz_pure_bs += np.sum(pure_active_sens_power) * self.g_xz_bs[self.now_step]
        g_yy_pure_bs += np.sum(pure_active_sens_power) * self.g_yy_bs[self.now_step]
        g_yz_pure_bs += np.sum(pure_active_sens_power) * self.g_yz_bs[self.now_step]
        g_zz_pure_bs += np.sum(pure_active_sens_power) * self.g_zz_bs[self.now_step]

        g_xx_pure_bs *= self.sens_para
        g_xy_pure_bs *= self.sens_para
        g_xz_pure_bs *= self.sens_para
        g_yy_pure_bs *= self.sens_para
        g_yz_pure_bs *= self.sens_para
        g_zz_pure_bs *= self.sens_para

        isac_active_sens_power = np.power(abs(self.active_sens_h[self.now_step] @ w_sens), 2)

        g_xx_isac_bs += np.sum(isac_active_sens_power) * self.g_xx_bs[self.now_step]
        g_xy_isac_bs += np.sum(isac_active_sens_power) * self.g_xy_bs[self.now_step]
        g_xz_isac_bs += np.sum(isac_active_sens_power) * self.g_xz_bs[self.now_step]
        g_yy_isac_bs += np.sum(isac_active_sens_power) * self.g_yy_bs[self.now_step]
        g_yz_isac_bs += np.sum(isac_active_sens_power) * self.g_yz_bs[self.now_step]
        g_zz_isac_bs += np.sum(isac_active_sens_power) * self.g_zz_bs[self.now_step]

        g_xx_isac_bs *= self.sens_para
        g_xy_isac_bs *= self.sens_para
        g_xz_isac_bs *= self.sens_para
        g_yy_isac_bs *= self.sens_para
        g_yz_isac_bs *= self.sens_para
        g_zz_isac_bs *= self.sens_para

        pure_passive_sens_power = np.power(abs(self.passive_sens_h[self.now_step] @ w_sens_pure), 2)

        g_xx_pure_comm += np.sum(pure_passive_sens_power) * self.g_xx_comm[self.now_step]
        g_xy_pure_comm += np.sum(pure_passive_sens_power) * self.g_xy_comm[self.now_step]
        g_xz_pure_comm += np.sum(pure_passive_sens_power) * self.g_xz_comm[self.now_step]
        g_yy_pure_comm += np.sum(pure_passive_sens_power) * self.g_yy_comm[self.now_step]
        g_yz_pure_comm += np.sum(pure_passive_sens_power) * self.g_yz_comm[self.now_step]
        g_zz_pure_comm += np.sum(pure_passive_sens_power) * self.g_zz_comm[self.now_step]

        g_xx_pure_comm *= self.sens_para
        g_xy_pure_comm *= self.sens_para
        g_xz_pure_comm *= self.sens_para
        g_yy_pure_comm *= self.sens_para
        g_yz_pure_comm *= self.sens_para
        g_zz_pure_comm *= self.sens_para

        g_xx_isac = g_xx_isac_bs
        g_xy_isac = g_xy_isac_bs
        g_xz_isac = g_xz_isac_bs
        g_yy_isac = g_yy_isac_bs
        g_yz_isac = g_yz_isac_bs
        g_zz_isac = g_zz_isac_bs

        if self.uav_assisted:
            g_xx_pure = g_xx_pure_bs + g_xx_pure_comm
            g_xy_pure = g_xy_pure_bs + g_xy_pure_comm
            g_xz_pure = g_xz_pure_bs + g_xz_pure_comm
            g_yy_pure = g_yy_pure_bs + g_yy_pure_comm
            g_yz_pure = g_yz_pure_bs + g_yz_pure_comm
            g_zz_pure = g_zz_pure_bs + g_zz_pure_comm
        else:
            g_xx_pure = g_xx_pure_bs
            g_xy_pure = g_xy_pure_bs
            g_xz_pure = g_xz_pure_bs
            g_yy_pure = g_yy_pure_bs
            g_yz_pure = g_yz_pure_bs
            g_zz_pure = g_zz_pure_bs

        gxx = np.zeros((self.T, 1), dtype=float)
        gxx[0:pure_sens_time, 0] = g_xx_pure
        gxx[pure_sens_time:, 0] = g_xx_isac

        gxy = np.zeros((self.T, 1), dtype=float)
        gxy[0:pure_sens_time, 0] = g_xy_pure
        gxy[pure_sens_time:, 0] = g_xy_isac

        gxz = np.zeros((self.T, 1), dtype=float)
        gxz[0:pure_sens_time, 0] = g_xz_pure
        gxz[pure_sens_time:, 0] = g_xz_isac

        gyy = np.zeros((self.T, 1), dtype=float)
        gyy[0:pure_sens_time, 0] = g_yy_pure
        gyy[pure_sens_time:, 0] = g_yy_isac

        gyz = np.zeros((self.T, 1), dtype=float)
        gyz[0:pure_sens_time, 0] = g_yz_pure
        gyz[pure_sens_time:, 0] = g_yz_isac

        gzz = np.zeros((self.T, 1), dtype=float)
        gzz[0:pure_sens_time, 0] = g_zz_pure
        gzz[pure_sens_time:, 0] = g_zz_isac

        # a = power.T @ (gyy @ gzz.T - gyz @ gyz.T) @ power
        # b = power.T @ (gxx @ gzz.T - gxz @ gxz.T) @ power
        # c = power.T @ (gxx @ gyy.T - gxy @ gxy.T) @ power
        #
        # d = power.T @ (gyy @ gzz.T - gyz @ gyz.T) @ power
        # e = power.T @ (gxy @ gzz.T - gxz @ gyz.T) @ power
        # f = power.T @ (gxy @ gyz.T - gxz @ gyy.T) @ power
        #
        # crb_a = a + b + c
        # crb_b = power.T @ (gxx @ a - gxy @ e + gxz @ f)
        #
        # CRB = crb_a / crb_b

        FIM = np.zeros((3, 3), dtype=float)
        FIM = np.array([[power.T @ gxx, power.T @ gxy, power.T @ gxz],
                        [power.T @ gxy, power.T @ gyy, power.T @ gyz],
                        [power.T @ gxy, power.T @ gyz, power.T @ gzz]], dtype=float)

        CRB = np.trace(np.linalg.inv(FIM))

        CRB *= 10000

        # if CRB < 0:
        #     print("1111111")

        return CRB

    def communication_performance(self):
        # comm beam
        w_comm = self.isac_beam_vector[:, 1].reshape(self.bs_tx_upa_num, 1)
        h_comm = self.comm_h[self.now_step]

        M = self.comm_uav_rx_upa_num
        N = self.bs_tx_upa_num

        # 线性 K（不是 dB）
        K = float(self.rician_factor)

        # 用与 LOS 相同的大尺度幅度来刻画散射项的尺度：
        # 由于 ||outer(a_r, a_t^H)||_F = 1，故 gain = ||H_LOS||_F
        gain = np.linalg.norm(h_comm, 'fro')

        # 生成复高斯散射项（单位方差，元素独立）
        H_sc = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(2)

        # 将大尺度路径损耗同样施加到散射项，并按 K 因子混合
        h_comm = np.sqrt(K / (K + 1)) * h_comm + np.sqrt(1 / (K + 1)) * (gain / np.sqrt(M * N)) * H_sc

        # sensing beam
        w_sens = self.isac_beam_vector[:, 0].T.reshape(self.bs_tx_upa_num, 1)
        h_passive_sens = self.passive_sens_h[self.now_step]

        # pure sensing beam
        w_sens_pure = self.pure_sens_beam_vector[:, 0].T.reshape(self.bs_tx_upa_num, 1)

        I = np.identity(self.comm_uav_rx_upa_num)
        R = (self.noise_power * I + h_comm @ w_sens @ w_sens.T.conjugate() @ h_comm.T.conjugate() +
             h_passive_sens @ w_sens @ w_sens.T.conjugate() @ h_passive_sens.T.conjugate() +
             h_passive_sens @ w_comm @ w_comm.T.conjugate() @ h_passive_sens.T.conjugate())
        R += 1e-9 * I

        self.comm_uav_sinr = w_comm.T.conjugate() @ h_comm.T.conjugate() @ np.linalg.inv(R) @ h_comm @ w_comm
        I_dk = np.identity(1)
        self.comm_uav_rate = self.band_width * np.log2(np.real(np.linalg.det(I_dk + self.comm_uav_sinr)))
        # sinr_db = 10 * np.log10(np.real(np.linalg.det(I_dk + self.comm_uav_sinr)))
        self.comm_uav_data = (self.T - self.beta_action) * self.comm_uav_rate / self.time
        return

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def init_sensing_vector(self):
        for i in range(self.total_step):
            known_uav_location = self.known_uav_location_init + (
                    self.known_uav_location_end - self.known_uav_location_init) * i / self.total_step
            self.known_uav_location_buffer[i] = known_uav_location / 500
            unknown_uav_location = self.unknown_uav_location_init + (
                    self.unknown_uav_location_end - self.unknown_uav_location_init) * i / self.total_step
            self.unknown_uav_location_buffer[i] = unknown_uav_location / 500

            # bs to comm uav for communication
            self.comm_h[i] = comm_channel_matrix(self.bs_location,
                                                 known_uav_location,
                                                 self.channel_comm_gain_1m,
                                                 self.bs_tx_upa_x_num, self.bs_tx_upa_y_num,
                                                 self.comm_uav_rx_upa_x_num, self.comm_uav_rx_upa_y_num)
            # bs to sensed uav to bs for active sensing
            self.active_sens_h[i] = sensing_channel_matrix(self.bs_location,
                                                           self.bs_location,
                                                           unknown_uav_location,
                                                           self.sens_uav_rcs,
                                                           self.channel_sens_gain_1m,
                                                           self.bs_tx_upa_x_num, self.bs_tx_upa_y_num,
                                                           self.bs_rx_upa_x_num, self.bs_rx_upa_y_num)
            # bs to sensed uav to communication for passive sensing
            self.passive_sens_h[i] = sensing_channel_matrix(self.bs_location,
                                                            known_uav_location,
                                                            unknown_uav_location,
                                                            self.sens_uav_rcs,
                                                            self.channel_sens_gain_1m,
                                                            self.bs_tx_upa_x_num, self.bs_tx_upa_y_num,
                                                            self.comm_uav_rx_upa_x_num, self.comm_uav_rx_upa_y_num)

            self.distance_bs_uav[i, 0] = np.linalg.norm(self.bs_location - unknown_uav_location)
            self.distance_bs_uav[i, 1] = np.linalg.norm(self.bs_location - known_uav_location)
            self.distance_uav_uav[i] = np.linalg.norm(unknown_uav_location - known_uav_location)

            # init sensing vector
            self.g_xx_bs[i] = ((-(self.bs_location[0] - unknown_uav_location[0]) / self.distance_bs_uav[i, 0] +
                                -(self.bs_location[0] - unknown_uav_location[0]) / self.distance_bs_uav[i, 0]) *
                               (-(self.bs_location[0] - unknown_uav_location[0]) / self.distance_bs_uav[i, 0] +
                                -(self.bs_location[0] - unknown_uav_location[0]) / self.distance_bs_uav[i, 0]))

            self.g_xy_bs[i] = ((-(self.bs_location[0] - unknown_uav_location[0]) / self.distance_bs_uav[i, 0] +
                                -(self.bs_location[0] - unknown_uav_location[0]) / self.distance_bs_uav[i, 0]) *
                               (-(self.bs_location[1] - unknown_uav_location[1]) / self.distance_bs_uav[i, 0] +
                                -(self.bs_location[1] - unknown_uav_location[1]) / self.distance_bs_uav[i, 0]))

            self.g_xz_bs[i] = ((-(self.bs_location[0] - unknown_uav_location[0]) / self.distance_bs_uav[i, 0] +
                                -(self.bs_location[0] - unknown_uav_location[0]) / self.distance_bs_uav[i, 0]) *
                               (-(self.bs_location[2] - unknown_uav_location[2]) / self.distance_bs_uav[i, 0] +
                                -(self.bs_location[2] - unknown_uav_location[2]) / self.distance_bs_uav[i, 0]))

            self.g_yy_bs[i] = ((-(self.bs_location[1] - unknown_uav_location[1]) / self.distance_bs_uav[i, 0] +
                                -(self.bs_location[1] - unknown_uav_location[1]) / self.distance_bs_uav[i, 0]) *
                               (-(self.bs_location[1] - unknown_uav_location[1]) / self.distance_bs_uav[i, 0] +
                                -(self.bs_location[1] - unknown_uav_location[1]) / self.distance_bs_uav[i, 0]))

            self.g_yz_bs[i] = ((-(self.bs_location[1] - unknown_uav_location[1]) / self.distance_bs_uav[i, 0] +
                                -(self.bs_location[1] - unknown_uav_location[1]) / self.distance_bs_uav[i, 0]) *
                               (-(self.bs_location[2] - unknown_uav_location[2]) / self.distance_bs_uav[i, 0] +
                                -(self.bs_location[2] - unknown_uav_location[2]) / self.distance_bs_uav[i, 0]))

            self.g_zz_bs[i] = ((-(self.bs_location[2] - unknown_uav_location[2]) / self.distance_bs_uav[i, 0] +
                                -(self.bs_location[2] - unknown_uav_location[2]) / self.distance_bs_uav[i, 0]) *
                               (-(self.bs_location[2] - unknown_uav_location[2]) / self.distance_bs_uav[i, 0] +
                                -(self.bs_location[2] - unknown_uav_location[2]) / self.distance_bs_uav[i, 0]))

            self.g_xx_comm[i] = ((-(self.bs_location[0] - unknown_uav_location[0]) / self.distance_bs_uav[i, 0] +
                                  -(known_uav_location[0] - unknown_uav_location[0]) / self.distance_uav_uav[i]) *
                                 (-(self.bs_location[0] - unknown_uav_location[0]) / self.distance_bs_uav[i, 0] +
                                  -(known_uav_location[0] - unknown_uav_location[0]) / self.distance_uav_uav[i]))

            self.g_xy_comm[i] = ((-(self.bs_location[0] - unknown_uav_location[0]) / self.distance_bs_uav[i, 0] +
                                  -(known_uav_location[0] - unknown_uav_location[0]) / self.distance_uav_uav[i]) *
                                 (-(self.bs_location[1] - unknown_uav_location[1]) / self.distance_bs_uav[i, 0] +
                                  -(known_uav_location[1] - unknown_uav_location[1]) / self.distance_uav_uav[i]))

            self.g_xz_comm[i] = ((-(self.bs_location[0] - unknown_uav_location[0]) / self.distance_bs_uav[i, 0] +
                                  -(known_uav_location[0] - unknown_uav_location[0]) / self.distance_uav_uav[i]) *
                                 (-(self.bs_location[2] - unknown_uav_location[2]) / self.distance_bs_uav[i, 0] +
                                  -(known_uav_location[2] - unknown_uav_location[2]) / self.distance_uav_uav[i]))

            self.g_yy_comm[i] = ((-(self.bs_location[1] - unknown_uav_location[1]) / self.distance_bs_uav[i, 0] +
                                  -(known_uav_location[1] - unknown_uav_location[1]) / self.distance_uav_uav[i]) *
                                 (-(self.bs_location[1] - unknown_uav_location[1]) / self.distance_bs_uav[i, 0] +
                                  -(known_uav_location[1] - unknown_uav_location[1]) / self.distance_uav_uav[i]))

            self.g_yz_comm[i] = ((-(self.bs_location[1] - unknown_uav_location[1]) / self.distance_bs_uav[i, 0] +
                                  -(known_uav_location[1] - unknown_uav_location[1]) / self.distance_uav_uav[i]) *
                                 (-(self.bs_location[2] - unknown_uav_location[2]) / self.distance_bs_uav[i, 0] +
                                  -(known_uav_location[2] - unknown_uav_location[2]) / self.distance_uav_uav[i]))

            self.g_zz_comm[i] = ((-(self.bs_location[2] - unknown_uav_location[2]) / self.distance_bs_uav[i, 0] +
                                  -(known_uav_location[2] - unknown_uav_location[2]) / self.distance_uav_uav[i]) *
                                 (-(self.bs_location[2] - unknown_uav_location[2]) / self.distance_bs_uav[i, 0] +
                                  -(known_uav_location[2] - unknown_uav_location[2]) / self.distance_uav_uav[i]))
