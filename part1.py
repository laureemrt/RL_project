# Imports
import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt

from resources.config import (
    config_highway
)
from dqn_utils import *


if __name__ == "__main__":
    # Create the environment
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    env.unwrapped.configure(config_highway)
    env.reset()
    obs, info = env.reset()

    action_space = env.action_space
    observation_space = env.observation_space

    gamma = 0.99
    batch_size = 32
    buffer_capacity = 10_000
    update_target_every = 32

    epsilon_start = 0.9
    decrease_epsilon_factor = 1000
    epsilon_min = 0.05

    learning_rate = 1e-2

    arguments = (action_space,
            observation_space,
            gamma,
            batch_size,
            buffer_capacity,
            update_target_every, 
            epsilon_start, 
            decrease_epsilon_factor, 
            epsilon_min,
            learning_rate,
        )

    # Create the model
    model = DQN(*arguments)

    N_episodes = 2000

    agent = DQN(*arguments)

        
    # Run the training loop
    losses = train(env, agent, N_episodes, eval_every=50)

    plt.plot(losses)

    # Evaluate the final policy
    rewards = eval_agent(agent, env, 20)
    print("")
    print("mean reward after training = ", np.mean(rewards))