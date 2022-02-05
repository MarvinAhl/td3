"""
Author: Marvin Ahlborn

This is an implementation of the twin-delayed
Deep Deterministic Policy Gradient (TD3) Algorithm using PyTorch.
"""

import torch
from torch import tensor
from torch import nn
import numpy as np

class PolicyNetwork(nn.Module):
    """
    The deterministic Policy Network
    """
    def __init__(self, state, actions, hidden=(512, 512)):
        """
        state: Integer telling the state dimension
        actions: Integer telling the number of actions
        hidden: Tuple of the each hidden layers nodes
        """
        super(PolicyNetwork, self).__init__()

        modules = []

        modules.append(nn.Linear(state, hidden[0]))
        modules.append(nn.LeakyReLU(0.1))

        for i in range(len(hidden) - 1):
            modules.append(nn.Linear(hidden[i], hidden[i+1]))
            modules.append(nn.LeakyReLU(0.1))
        
        modules.append(nn.Linear(hidden[-1], actions))
        modules.append(nn.Tanh())  # Action Output is float between -1 and 1

        self.module_stack = nn.Sequential(*modules)
    
    def forward(self, state):
        actions = self.module_stack(state)
        return actions

class ValueNetwork(nn.Module):
    """
    Twin Q-Network
    """
    def __init__(self, state, actions, hidden=(512, 512)):
        """
        state: Integer telling the state dimension
        actions: Integer telling the number of actions
        hidden: Tuple of the each hidden layers nodes
        """
        super(ValueNetwork, self).__init__()

        modules_a = []
        modules_b = []

        modules_a.append(nn.Linear(state + actions, hidden[0]))  # Takes in State and Action
        modules_b.append(nn.Linear(state + actions, hidden[0]))
        modules_a.append(nn.LeakyReLU(0.1))
        modules_b.append(nn.LeakyReLU(0.1))

        for i in range(len(hidden) - 1):
            modules_a.append(nn.Linear(hidden[i], hidden[i+1]))
            modules_b.append(nn.Linear(hidden[i], hidden[i+1]))
            modules_a.append(nn.LeakyReLU(0.1))
            modules_b.append(nn.LeakyReLU(0.1))
        
        modules_a.append(nn.Linear(hidden[-1], 1))  # Only one output for the State-Action-Value
        modules_b.append(nn.Linear(hidden[-1], 1))

        self.module_stack_a = nn.Sequential(*modules_a)
        self.module_stack_b = nn.Sequential(*modules_b)
    
    def forward(self, state, actions):
        state_action_input = torch.cat((state, actions), dim=-1)
        value_output_a = self.module_stack_a(state_action_input)
        value_output_b = self.module_stack_b(state_action_input)
        return value_output_a, value_output_b

class ReplayBuffer:
    """
    Uniformly random Replay Buffer
    """
    def __init__(self, state, actions, max_len=50000):
        """
        state: Integer of State Dimension
        actions: Integer of Number of Actions
        """
        self.states = np.empty((max_len, state), dtype=np.float32)
        self.actions = np.empty((max_len, actions), dtype=np.float32)
        self.rewards = np.empty(max_len, dtype=np.float32)
        self.next_states = np.empty((max_len, state), dtype=np.float32)
        self.terminals = np.empty(max_len, dtype=np.int8)

        self.index = 0
        self.full = False
        self.max_len = max_len
        self.rng = np.random.default_rng()
    
    def store_experience(self, state, actions, reward, next_state, terminal):
        """
        Stores given SARS Experience in the Replay Buffer.
        Returns True if the last element has been written to memory and the
        Buffer will start over replacing the first elements at next call.
        """
        self.states[self.index] = state
        self.actions[self.index] = actions
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.terminals[self.index] = terminal
        
        self.index += 1
        self.index %= self.max_len  # Replace oldest Experiences if Buffer is full

        if self.index == 0:
            self.full = True
            return True
        return False

    def get_experiences(self, batch_size):
        """
        Returns batch of experiences for replay.
        """
        indices = self.rng.choice(self.__len__(), batch_size)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        terminals = self.terminals[indices]

        return states, actions, rewards, next_states, terminals

    def __len__(self):
        return self.max_len if self.full else self.index

class TD3:
    def __init__(self, state, actions, explore_factors=None, policy_hidden=(512, 512), value_hidden=(512, 512), gamma=0.99,
                 learning_rate=0.0002, explore_start=1.0, explore_decay_steps=20000, explore_min=0.05, smoothing_noise=0.05, noise_clip=0.1,
                 buffer_size_max=50000, buffer_size_min=1024, batch_size=64, replays=1, policy_update_every=2, tau=0.01, device='cpu'):
        """
        state: Integer of State Dimension
        actions: Integer of Number of Action
        """
        self.state = state
        self.actions = actions

        self.policy_hidden = policy_hidden
        self.policy_net = PolicyNetwork(state, actions, policy_hidden).to(device)
        self.target_policy_net = PolicyNetwork(state, actions, policy_hidden).to(device)
        
        self.value_hidden = value_hidden
        self.value_net = ValueNetwork(state, actions, value_hidden).to(device)
        self.target_value_net = ValueNetwork(state, actions, value_hidden).to(device)

        self._update_targets(1.0)  # Fully copy Online Net weights to Target Net
  
        # Using RMSProp because it's more stable, not as aggressive than Adam
        self.policy_optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.RMSprop(self.value_net.parameters(), lr=learning_rate)

        self.buffer = ReplayBuffer(state, actions, buffer_size_max)
        self.buffer_size_max = buffer_size_max
        self.buffer_size_min = buffer_size_min
        self.batch_size = batch_size
        self.replays = replays  # On how many batches it should train after each step

        # Can be calculated by exp(- dt / lookahead_horizon)
        self.gamma = gamma  # Reward discount rate

        self.learning_rate = learning_rate

        self.n_q_update = 0
        self.policy_update_every = policy_update_every

        self.smoothing_noise = smoothing_noise  # TD3's action noise to make network generalize better
        self.noise_clip = noise_clip

        self.rng = np.random.default_rng()

        if not explore_factors == None:
            self.explore_factors = np.array(explore_factors, dtype=np.float32)
        else:
            self.explore_factors = np.ones(actions, dtype=np.float32)

        # Linearly decay exploration rate from start to min in a given amount of steps
        self.explore_rate = explore_start
        self.explore_start = explore_start
        self.explore_decay = (explore_start - explore_min) / explore_decay_steps
        self.explore_min = explore_min

        self.tau = tau  # Mixing parameter for polyak averaging of target and online network

        self.device = device
    
    def reset(self):
        """
        Reset object to its initial state if you want to do multiple training passes with it
        """
        self.policy_net = PolicyNetwork(self.state, self.actions, self.policy_hidden).to(self.device)
        self.target_policy_net = PolicyNetwork(self.state, self.actions, self.policy_hidden).to(self.device)
        self.value_net = ValueNetwork(self.state, self.actions, self.value_hidden).to(self.device)
        self.target_value_net = ValueNetwork(self.state, self.actions, self.value_hidden).to(self.device)
        self._update_targets(1.0)  # Fully copy Online Net weights to Target Net

        self.policy_optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.RMSprop(self.value_net.parameters(), lr=self.learning_rate)

        self.buffer = ReplayBuffer(self.state, self.actions, self.buffer_size_max)

        self.explore_rate = self.explore_start

        self.rng = np.random.default_rng()

    def act(self, state):
        """
        Decides on action based on current state using Policy Net.
        """
        with torch.no_grad():
            state = tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            actions = self.policy_net(state).squeeze().cpu().numpy()

        return actions
    
    def act_explore(self, state):
        """
        Decides on action but adds Gaussian Noise for exploratory purposes.
        """
        actions = self.act(state)  # Optimal actions
        explore_actions = self.rng.normal(actions, self.explore_factors * self.explore_rate)
        explore_actions_clip = np.clip(explore_actions, -1.0, 1.0)

        self._update_parameters()

        return explore_actions_clip
    
    def experience(self, state, actions, reward, next_state, terminal):
        """
        Takes experience and stores it for replay.
        """
        self.buffer.store_experience(state, actions, reward, next_state, terminal)
    
    def train(self):
        """
        Train Q-Network on a batch from the replay buffer.
        """
        if len(self.buffer) < self.buffer_size_min:
            return  # Dont train until Replay Buffer has collected a certain number of initial experiences

        for _ in range(self.replays):
            states, actions, rewards, next_states, terminals = self.buffer.get_experiences(self.batch_size)

            states = torch.from_numpy(states).to(self.device)
            rewards = torch.from_numpy(rewards).to(self.device)
            actions = torch.from_numpy(actions).to(self.device)
            next_states = torch.from_numpy(next_states).to(self.device)
            terminals = torch.from_numpy(terminals).to(self.device)

            # Q-Function training
            next_actions = self.target_policy_net(next_states)
            smooth_noise = torch.normal(0.0, self.smoothing_noise, size=next_actions.shape, device=self.device)
            smooth_noise_clip = torch.clip(smooth_noise, -self.noise_clip, self.noise_clip)
            smooth_actions = next_actions + smooth_noise_clip
            smooth_actions_clip = torch.clip(smooth_actions, -1.0, 1.0)
            next_values_a, next_values_b = self.target_value_net(next_states, smooth_actions_clip)

            td_targets = rewards + self.gamma * torch.min(next_values_a, next_values_b).detach() * (1 - terminals)
            predictions_a, predictions_b = self.value_net(states, actions)
            td_errors_a = td_targets - predictions_a
            td_errors_b = td_targets - predictions_b
            loss = td_errors_a.pow(2).mean() + td_errors_b.pow(2).mean()

            self.value_optimizer.zero_grad()
            loss.backward()
            self.value_optimizer.step()

            self.n_q_update += 1
            self.n_q_update %= self.policy_update_every

            # Policy Function training every few Q-Updates
            if self.n_q_update == 0:
                pred_actions = self.policy_net(states)
                pred_values_a, _ = self.value_net(states, pred_actions)
                loss = -pred_values_a.mean()  # Only use Stream a to update policy

                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()

        self._update_targets(self.tau)
    
    def save_net(self, path):
        torch.save(self.policy_net.state_dict(), 'policy_' + path)
        torch.save(self.value_net.state_dict(), 'value_' + path)
    
    def load_net(self, path):
        self.policy_net.load_state_dict(torch.load('policy_' + path))
        self.value_net.load_state_dict(torch.load('value_' + path))
        self._update_targets(1.0)  # Also load weights into target net
    
    def _update_targets(self, tau):
        """
        Update Target Networks by blending Target und Online Network weights using the factor tau (Polyak Averaging)
        A tau of 1 just copies the whole online network over to the target network
        """
        for policy_param, target_policy_param in zip(self.policy_net.parameters(), self.target_policy_net.parameters()):
            target_policy_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_policy_param.data)
        
        for value_param, target_value_param in zip(self.value_net.parameters(), self.target_value_net.parameters()):
            target_value_param.data.copy_(tau * value_param.data + (1.0 - tau) * target_value_param.data)
    
    def _update_parameters(self):
        """
        Decay epsilon
        """
        self.explore_rate -= self.explore_decay
        self.explore_rate = max(self.explore_rate, self.explore_min)