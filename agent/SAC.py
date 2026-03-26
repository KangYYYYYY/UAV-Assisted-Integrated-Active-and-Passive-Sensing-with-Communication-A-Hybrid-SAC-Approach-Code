import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.distributions import Normal
from env.MyEnvPara_HPSAC import MyEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std_layer = nn.Linear(hidden_width, action_dim)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # We output the log_std to ensure that std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)  # Generate a Gaussian distribution
        if deterministic:  # When evaluating，we use the deterministic policy
            a = mean
        else:
            a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)

        if with_logprob:  # The method refers to Open AI Spinning up, which is more stable.
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        a = self.max_action * (torch.tanh(a)) / 2  # Use tanh to compress the unbounded Gaussian distribution into a bounded action interval.

        return a, log_pi


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, env_args):
        self.max_action = max_action
        self.hidden_width = 1024  # The number of neurons in hidden layers of the neural network
        self.batch_size = 1024  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = env_args["learning_rate"]  # learning rate
        self.adaptive_alpha = True  # Whether to automatically learn the temperature alpha
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -0.5 *action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, requires_grad=False).to(device)
            self.log_alpha.requires_grad = True
            self.alpha = self.log_alpha.exp().to(device)
            self.alpha = 0.6
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = 0.2

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action).to(device)
        self.critic = Critic(state_dim, action_dim, self.hidden_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(device)
        a, _ = self.actor(s, deterministic, False)  # When choosing actions, we do not need to compute log_pi
        return a.data.cpu().numpy().flatten()

    def learn(self, relay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

        # TODO soft state objective
        with torch.no_grad():
            batch_a_, log_pi_ = self.actor(batch_s_)  # a' from the current policy
            # Compute target Q
            target_Q1, target_Q2 = self.critic_target(batch_s_, batch_a_)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * (torch.min(target_Q1, target_Q2) - self.alpha * log_pi_)

        # Compute current Q
        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute actor loss
        a, log_pi = self.actor(batch_s)
        Q1, Q2 = self.critic(batch_s, a)
        Q = torch.min(Q1, Q2)
        # TODO policy objective
        actor_loss = (self.alpha * log_pi - Q).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # TODO temperature objective
        # Update alpha
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Softly update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        return actor_loss.data.cpu().numpy(), critic_loss.data.cpu().numpy()


def evaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        while episode_steps < env.total_step:
            episode_steps += 1
            a = agent.choose_action(s, deterministic=True)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward
    return int(evaluate_reward / times)


def reward_adapter(r, env_index):
    if env_index == 0:  # Pendulum-v1
        r = (r + 8) / 8
    elif env_index == 1:  # BipedalWalker-v3
        if r <= -100:
            r = -1
    return r

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(1e6)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float).to(device)
        batch_a = torch.tensor(self.a[index], dtype=torch.float).to(device)
        batch_r = torch.tensor(self.r[index], dtype=torch.float).to(device)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float).to(device)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float).to(device)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw

if __name__ == '__main__':
    env = MyEnv()
    env_evaluate = MyEnv()
    for select_model in range(1):
        for num in range(10):
            print("select_model={}".format(select_model))
            print("num={}".format(num))
            number = 1

            s = env.reset()
            # Set random seed
            seed = num
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

            max_action = float(env.action_space.high[0])
            dis_action_max = env.dis_action_max

            max_episode_steps = env.total_step  # Maximum number of steps per episode
            env.max_episode_steps = max_episode_steps
            max_episode_steps = env.max_episode_steps  # Maximum number of steps per episode

            agent = SAC(state_dim, action_dim, max_action)

            replay_buffer = ReplayBuffer(state_dim, action_dim)

            max_train_steps = 8000  # Maximum number of training steps
            random_steps = 2000  # Take the random actions in the beginning for the better exploration
            evaluate_freq = 1  # Evaluate the policy every 'evaluate_freq' steps
            evaluate_num = 0  # Record the number of evaluations
            evaluate_rewards = []  # Record the rewards during the evaluating
            actor_losses = []
            critic_losses = []
            total_steps = 0  # Record the total steps during the training
            rewards = []
            episode_num = 0
            while total_steps < max_train_steps:
                s = env.reset()
                episode_steps = 0
                evaluate_reward = 0
                done = False
                episode_num += 1
                while episode_steps < max_episode_steps:
                    episode_steps += 1
                    if total_steps < random_steps:  # Take the random actions in the beginning for the better exploration
                        a = env.action_space.sample()
                        a[0] = int(a[0] * dis_action_max)
                    else:
                        a = agent.choose_action(s)
                        a[0] = int(a[0] * dis_action_max)
                    s_, r_, done_, _ = env.step(a)

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

                    if total_steps >= random_steps:
                        actor_loss, critic_loss = agent.learn(replay_buffer)
                        actor_losses.append(actor_loss)
                        critic_losses.append(critic_loss)
                        # evaluate_reward = evaluate_policy(env_evaluate, agent)
                        # evaluate_rewards.append(evaluate_reward)
                        # print("num:{} \t reward:{}".format(total_steps, evaluate_reward))

                    # Evaluate the policy every 'evaluate_freq' steps
                    if (total_steps + 1) % evaluate_freq == 0 and total_steps >= random_steps:
                        evaluate_num += 1
                        evaluate_reward = evaluate_policy(env_evaluate, agent)
                        evaluate_rewards.append(evaluate_reward)
                        print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))

                    total_steps += 1
                # evaluate_num += 1
                # evaluate_rewards.append(evaluate_reward)
                # print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))

            # TODO Save Network Paras
            # torch.save(agent.actor_1.state_dict(), 'dqn_maze_model.pkl')
            # self.actor_1.load_state_dict(torch.load('dqn_maze_model.pkl'))  # 加载模型

            np.save('data/SAC_reward_{}.npy'.format(num),
                    np.array(evaluate_rewards))
            np.save('data/SAC_actor_loss_{}.npy'.format(num), np.array(actor_losses))
            np.save('data/SAC_critic_loss_{}.npy'.format(num), np.array(critic_losses))
            replay_buffer = None