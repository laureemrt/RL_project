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
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy

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
            action = env.action_space.sample() #agent.get_action(state)
            # print("action", action)
            state, reward, terminated, _, info = env_copy.step(action)
            reward_sum += reward
            done = terminated  or info['crashed'] or info['rewards']['collision_reward'] or not info['rewards']['on_road_reward'] #or truncated
        episode_rewards[i] = reward_sum
    return episode_rewards


def run_one_episode(env, agent, display=True):
    display_env = deepcopy(env)
    done = False
    state, _ = display_env.reset()

    rewards = 0

    while not done:
        action = agent.action_space.sample()
        # print(f"Action: {action}")
        next_state, reward, done, terminated, info = display_env.step(action)
        # print(f"State: {state}, Reward: {reward}, Done: {done}")
        rewards += reward
        state = next_state
        done = terminated or info['crashed'] or info['rewards']['collision_reward'] or not info['rewards']['on_road_reward']
        if display: 
            clear_output(wait=True)
            plt.imshow(display_env.render())
            plt.show()
    if display:
        display_env.close()
    print(f'Episode length {rewards}')


def train(env, agent, n_episodes, eval_every=100, reward_threshold=300, n_eval=10):
    total_time = 0
    for ep in range(n_episodes):
        done = False
        state, _ = env.reset()
        while not done: 
            action = agent.action_space.sample() # agent.get_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            # print("Info", info)
            # if not info['rewards']['on_road_reward']: # If the car is off the road
            #     reward -= 1
            # else:
            #     reward += 0.01 # Reward for staying on the road
            agent.update(state, action, reward, terminated, next_state)

            state = next_state

            done = terminated or truncated or info['crashed'] == True # or info['rewards']['collision_reward'] or not info['rewards']['on_road_reward']
            total_time += 1
            # if ep == n_episodes -1 :
            #     clear_output(wait=True)
            #     plt.imshow(env.render())
            #     plt.show()
            #     env.close()

        if ((ep+1)% eval_every == 0):
            mean_reward = np.mean(eval_agent(agent, env, n_sim=n_eval))
            print("episode =", ep+1, ", mean reward = ", mean_reward)
            if mean_reward >= reward_threshold:
                print(f"Solved in {ep} episodes!")
                break
    print("Finished training")
              
    return 

def resize(x, n_action):
    # Resize the action to [-1, 1]
    return x*2/n_action -1

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
        # print("obs_size", obs_size, "hidden_size", hidden_size, "n_actions", n_actions)

    def forward(self, x):
        # print("x", x.shape)
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
    
    # def get_action(self, state, epsilon=None):
    #     state_tensor = torch.tensor(state).unsqueeze(0)
    #     with torch.no_grad():
    #         unn_log_probs = self.policy_net.forward(state_tensor).numpy()[0]
    #         p = np.exp(unn_log_probs - np.min(unn_log_probs))
    #         p = p / np.sum(p)
    #         return np.random.choice(np.arange(self.action_space.shape[0]), p=p)

    def get_action(self, state):
        # print(state)
        flat_state = torch.tensor(state['observation'].flatten()).float() #torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.policy_net.forward(flat_state)
            # Get the action with the highest probability
            action = action_probs.argmax()
            action = resize(action, 21)
            action = action.item() # output is a tensor of size 1
        return [action]


    def reset(self):
        hidden_size = 128 # 128
        obs_size = self.observation_space["observation"].shape[0]
        n_actions = self.action_space.shape[0]

        # Observation_space = Box(-inf, inf, (2,12,12))
        # Action space = Box(-1, 1, (1,))

        # obs_size = 288 # self.observation_space.shape = 2*12*12
        # n_actions = 21 # self.action_space.shape

        self.policy_net = Net(obs_size, hidden_size, n_actions)

        self.scores = []
        self.current_episode = []

        self.optimizer = optim.Adam(
            params=self.policy_net.parameters(), lr=self.learning_rate
        )

        self.n_eps = 0


class Reinforce(ReinforceSkeleton):
    
    def _gradient_returns(self, rewards, gamma):
        """
        Turns a list of rewards into the list of returns * gamma**t
        """
        G = 0
        returns_list = []
        T = len(rewards)
        full_gamma = np.power(gamma, T)
        for t in range(T):
            G = rewards[T-t-1] + gamma * G
            full_gamma = full_gamma / gamma
            returns_list.append(full_gamma * G)
        return torch.tensor(returns_list[::-1], dtype=torch.float32)

    # def update(self, state, action, reward, terminated, next_state):
    #     # print(state)
    #     self.current_episode.append((
    #         torch.tensor(state["observation"], dtype=torch.float32).unsqueeze(0),
    #         torch.tensor([[action]], dtype=torch.float32),
    #         torch.tensor([reward], dtype=torch.float32),
    #     )
    #     )
    #     # print(self.current_episode)

    #     if terminated: 
    #         self.n_eps += 1

    #         states, actions, rewards = tuple(
    #             [torch.cat(data) for data in zip(*self.current_episode)]
    #         )

    #         current_episode_returns = self._gradient_returns(rewards, self.gamma)

    #         unn_log_probs = self.policy_net.forward(states[0])
   
    #         # log_probs = unn_log_probs - torch.log(torch.sum(torch.exp(unn_log_probs), dim=-1)).unsqueeze(1)
    #         log_probs = unn_log_probs - torch.log(torch.sum(torch.exp(unn_log_probs), dim=-1)).unsqueeze(0)
    #         print(actions[0][0])
    #         print(log_probs)
    #         # print(log_probs.gather(1, actions[0][0]))

    #         full_neg_score = - torch.dot(log_probs.gather(0, actions[0][0]).squeeze(), current_episode_returns).unsqueeze(0)#).sum()

    #         self.current_episode = []

    #         self.optimizer.zero_grad()
    #         full_neg_score.backward()
    #         self.optimizer.step()

    def update(self, state, action, reward, done, next_state):

        self.current_episode.append((state, action, reward))
        if done:
            states, actions, rewards = zip(*self.current_episode)
            returns = self._gradient_returns(rewards, self.gamma)
            flat_states = torch.tensor(states.flatten()).float()
            returns = (returns - returns.mean())/ (returns.std() + 1e-9)
            action_probs = self.policy_net.forward(flat_states)
            action_probs = action_probs.item() # action_probs.gather(1, actions.view(-1, 1)).squeeze()
            loss = -torch.log(action_probs)*returns
            # loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.current_episode = []
        return

    
class ReinforceBatch(Reinforce):
    def update(self, state, action, reward, terminated, next_state):
        #torch.tensor(state).unsqueeze(0),
        self.current_episode.append((
            torch.tensor(state['observation'].flatten()).float(),
            torch.tensor([[action]], dtype=torch.float32),
            torch.tensor([reward]),
        )
        )

        if terminated:
            self.n_eps += 1

            states, actions, rewards = tuple(
                [torch.cat(data) for data in zip(*self.current_episode)]
            )

            current_episode_returns = self._gradient_returns(rewards, self.gamma)

            unn_log_probs = self.policy_net.forward(states)
            log_probs = unn_log_probs - torch.log(torch.sum(torch.exp(unn_log_probs), dim=0)).unsqueeze(0)
            # print(log_probs, actions)

            # Adjusting indices to match the shape of log_probs
            indices = actions.squeeze(0).squeeze(0).numpy().tolist()  # Squeeze to remove the extra dimensions
            selected_probs = log_probs[indices]  # Using basic indexing to select elements from log_probs
            
            # print(selected_probs, current_episode_returns)
            score = torch.dot(selected_probs, current_episode_returns.repeat(2)).unsqueeze(0)
            self.scores.append(score)

            # self.scores.append(torch.dot(log_probs.gather(1, actions.squeeze(0).squeeze(0)).squeeze(), current_episode_returns).unsqueeze(0))
            self.current_episode = []

            if (self.n_eps % self.episode_batch_size)==0:
                self.optimizer.zero_grad()
                full_neg_score = - torch.cat(self.scores).sum() / self.episode_batch_size
                full_neg_score.backward()
                self.optimizer.step()
                
                self.scores = []
