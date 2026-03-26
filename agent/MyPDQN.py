import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        a = self.max_action * torch.tanh(self.l3(s))  # [-max,max]
        return a


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, para_action_dim, dis_action_dim, hidden_width):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + para_action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, dis_action_dim)

    def forward(self, s, a):
        q = F.relu(self.l1(torch.cat([s, a], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class PDQN(object):
    def __init__(self, state_dim, para_action_dim, dis_action_dim, max_action, env_args):
        self.max_action = max_action
        self.hidden_width = 1024  # The number of neurons in hidden layers of the neural network
        self.batch_size = 1024  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = env_args["learning_rate"]  # learning rate
        # self.lr = 0.0001 * 10 ** learning_rate # learning rate

        self.VAR = dis_action_dim / 2 # control exploration
        self.GATE = 1  # dis_action gate
        self.noise_std = 0.1 * max_action
        self.dis_action_dim = dis_action_dim

        self.actor = Actor(state_dim, para_action_dim, self.hidden_width, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, para_action_dim, dis_action_dim, self.hidden_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.MseLoss = nn.MSELoss()


    def choose_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(device)
        para_a = self.actor(s)  # When choosing actions, we do not need to compute log_pi
        dis_a_1 = self.critic(s, para_a)

        dis_a = dis_a_1.data.cpu().numpy()

        dis_a = np.argmax(dis_a, axis=1).clip(1, self.dis_action_dim)

        para_a = para_a.data.cpu().numpy().clip(-1 + 0.0000001, self.max_action-0.0000001)

        # add exploration
        if np.random.rand() < self.GATE and deterministic == False:
            add_noise = np.random.normal(dis_a, self.VAR)
            index_noise = int(np.mod(np.floor(add_noise), self.dis_action_dim).clip(1, self.dis_action_dim))
            # 0, ...,6 ----- 1, ..., 7
            dis_a = np.array([index_noise])
            para_a = np.random.normal(para_a, self.noise_std).clip(-1 + 0.0000001, self.max_action-0.0000001)


        a = np.concatenate((dis_a, para_a.flatten()), axis=0)
        return a

    def learn(self, relay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

        with torch.no_grad():
            batch_a_ = self.actor(batch_s_)  # a' from the current policy
            # Compute target Q
            Q_ = self.critic_target(batch_s_, batch_a_)
            Q_max = torch.max(Q_, 1, keepdim=True)[0]
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * (Q_max)

        # Compute current Q
        current_Q = self.critic(batch_s, batch_a[:, 1:])

        int_tensor = batch_a[:, 0].int().view(-1, 1).long()
        current_Q1 = current_Q.gather(1, int_tensor)

        # Compute critic loss
        critic_loss = self.MseLoss(target_Q, current_Q1)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute actor loss
        a = self.actor(batch_s)
        Q1= self.critic(batch_s, a)
        Q11 = torch.max(Q1, 1, keepdim=True)[0]
        actor_loss = -Q11.mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        return actor_loss.data.cpu().numpy(), critic_loss.data.cpu().numpy()
