import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        states, actions, rewards, next_states, terminated = zip(*random.sample(self.memory, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminated)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    """
    Basic neural net.
    """

    def __init__(self, obs_size, archi, n_actions):
        super(Net, self).__init__()
        nb_hidden_layers = len(archi)

        if nb_hidden_layers > 0:
            modules = [nn.Linear(obs_size, archi[0]), nn.ReLU()]
        else:
            modules = []

        for idx in range(len(archi) - 1):
            modules.append(nn.Linear(archi[idx], archi[idx + 1]))
            modules.append(nn.ReLU())

        if n_actions > 0:
            last_layer_dim = archi[-1] if nb_hidden_layers > 0 else obs_size
            modules.append(nn.Linear(last_layer_dim, n_actions))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x.view(x.shape[0],-1))


class DQN:
    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        batch_size,
        buffer_capacity,
        update_target_every,
        epsilon_start,
        epsilon_decay,
        epsilon_min,
        learning_rate,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma

        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.update_target_every = update_target_every

        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.learning_rate = learning_rate
        self.n_steps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reset()

    def get_q(self, state):
        """
        Compute Q function for a states
        """
        state_tensor = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            output = self.q_net.forward(state_tensor) # shape (1,  n_actions)
        return output.numpy()[0]  # shape  (n_actions)
 
    def update(self, terminated):
        # add data to replay buffer

        if len(self.buffer) < self.batch_size:
            return np.inf

        # get batch
        """transitions = self.buffer.sample(self.batch_size)"""
        device = self.device
        state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = self.buffer.sample(self.batch_size)
        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(device)
        action_batch = torch.tensor(action_batch, dtype=torch.long).to(device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).to(device)
        terminated_batch = torch.tensor(terminated_batch, dtype=torch.uint8).to(device)

        values = self.q_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute the ideal Q values
        with torch.no_grad():
            next_state_actions = self.q_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_state_values = (1 - terminated_batch) * self.target_net(next_state_batch).gather(1, next_state_actions).squeeze(1)
            targets = next_state_values * self.gamma + reward_batch

        loss = self.loss_function(values, targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if not ((self.n_steps + 1) % self.update_target_every):
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.decrease_epsilon()

        self.n_steps += 1
        if terminated:
            self.n_eps += 1
    
    def decrease_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
       
    def get_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.q_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(9)]], dtype=torch.long)
        
    def calculate_reward(self, env, info, reward):
        head = env.unwrapped.vehicle.heading
        if (head > np.pi/2 and head < 3*np.pi/2) or (head < -np.pi/2 and head > -3*np.pi/2):
            reward = -0.2

        on_road_reward = info['rewards']['on_road_reward']
        speed_reward = info['speed']
        collision_reward = info['rewards']['collision_reward']


        if float(speed_reward) <= 20 :
            reward = -100
        
        if collision_reward < 0:
            reward = -200

        return reward, on_road_reward, speed_reward, collision_reward
    

    def reset(self):
        archi = [256, 256]

        obs_size = self.observation_space.shape[0] * self.observation_space.shape[1] * self.observation_space.shape[2]
        n_actions = 9

        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.q_net = Net(obs_size, archi, n_actions)
        self.target_net = Net(obs_size, archi, n_actions)

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(
            params=self.q_net.parameters(), lr=self.learning_rate
        )

        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0


def train(env, agent:DQN, N_episodes, eval_every=100):
    state, _ = env.reset()
    losses = []
    time_step_reward = 0.01
    offRoad = 0
    totalRewardList = []
    speedAverageList = []

    for ep in range(N_episodes):
        total_reward = 0
        agent.n_steps = 0
        speedAverage = 0
        done = False
        state, _ = env.reset()
        state = state.flatten()
        while not done:
            agent.n_steps += 1
            state_tensor = torch.tensor([state], dtype=torch.float32).to(agent.device)
            action = agent.get_action(state_tensor).item()
            next_state, reward, done, truncated, info = env.step(action)

            if info['rewards']['on_road_reward'] <= 0:
                offRoad += 1
            next_state = next_state.flatten()

            reward, on_road_reward, speed_reward, collision_reward = agent.calculate_reward(env, info, reward)
            reward += agent.n_steps * time_step_reward
            total_reward += reward
            terminated = done or truncated
            agent.buffer.push(state, action, reward, next_state, terminated)

            if on_road_reward > 0:
                agent.update(terminated)
                state = next_state
                
            if speed_reward <= 0 or collision_reward < 0:
                break
            if done or truncated:
                break
            
            speedAverage += float(info['speed'])
        
        speedAverageList.append(speedAverage/agent.n_steps)
        totalSpeedAverage = np.mean(speedAverageList[-99:])
        totalRewardList.append(total_reward)
        losses.append(total_reward)
                
        if ((ep+1)% eval_every == 0):
            print("episode =", ep+1, ", mean reward (last 100 ep) = ", np.mean(totalRewardList))
            totalRewardList = []
            print(f"Average speed : {totalSpeedAverage}")
            print()

    return losses, speedAverageList