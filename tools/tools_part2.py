"""
Tools module containing Part2 classes and functions.
"""

######################
### Python imports ###
######################

from copy import deepcopy
import numpy as np
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
from IPython.display import clear_output
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
import gymnasium as gym



class RandomAgent: 
    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        return
    
    def get_action(self, state, **kwargs):
        return self.action_space.sample()
    
    def update(self, *data):
        pass


def eval_agent(agent, env, n_sim=10):
    """    
    Monte Carlo evaluation of the agent.

    Repeat n_sim times:
        * Run the agent policy until the environment reaches a terminal state (= one episode)
        * Compute the sum of rewards in this episode
        * Store the sum of rewards in the episode_rewards array.
    """
    env_copy = deepcopy(env)
    episode_rewards = np.zeros(n_sim)
    for i in range(n_sim):
        state, _ = env_copy.reset()
        reward_sum = 0
        done = False
        while not done: 
            action = agent.get_action(state, epsilon=0)
            state, reward, terminated, truncated, _ = env_copy.step(action)
            reward_sum += reward
            done = terminated or truncated
        episode_rewards[i] = reward_sum
    return episode_rewards


def run_one_episode(env, agent, display=True):
    display_env = deepcopy(env)
    done = False
    state, _ = display_env.reset()

    rewards = 0

    while not done:
        action = agent.action_space.sample() # agent.get_action(state, display_env) #epsilon=0)
        print(f"Action: {action}")
        state, reward, done, _, _ = display_env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}")
        rewards += reward
        if display: 
            clear_output(wait=True)
            plt.imshow(display_env.render())
            plt.show()
    if display:
        display_env.close()
    print(f'Episode length {rewards}')


def train(env, agent, N_episodes, eval_every=100, reward_threshold=300, n_eval=10):
    total_time = 0
    for ep in range(N_episodes):
        done = False
        state, _ = env.reset()
        while not done: 
            action = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, terminated, next_state)

            state = next_state

            done = terminated or truncated
            total_time += 1

        if ((ep+1)% eval_every == 0):
            mean_reward = np.mean(eval_agent(agent, env, n_sim=n_eval))
            print("episode =", ep+1, ", reward = ", mean_reward)
            if mean_reward >= reward_threshold:
                break
                
    return 

class Net(nn.Module):
    """
    Basic neural net.
    """

    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        print("obs_size", obs_size, "hidden_size", hidden_size, "n_actions", n_actions)

    def forward(self, x):
        return self.net(x)
    

class ReinforceSkeleton:
    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        episode_batch_size,
        learning_rate,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma

        self.episode_batch_size = episode_batch_size
        self.learning_rate = learning_rate

        self.reset()

    def update(self, state, action, reward, terminated, next_state):
        pass
    
    def get_action(self, state, epsilon=None):
        state_tensor = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            unn_log_probs = self.policy_net.forward(state_tensor).numpy()[0]
            p = np.exp(unn_log_probs - np.min(unn_log_probs))
            p = p / np.sum(p)
            return np.random.choice(np.arange(self.action_space.shape[0])) # , p=p


    def reset(self):
        hidden_size = 128 # 128

        obs_size = 12 # self.observation_space.shape[0]
        n_actions = self.action_space.shape[0]

        self.policy_net = Net(obs_size, hidden_size, n_actions)

        self.scores = []
        self.current_episode = []

        self.optimizer = optim.Adam(
            params=self.policy_net.parameters(), lr=self.learning_rate
        )

        self.n_eps = 0