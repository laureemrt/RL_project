# Imports
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import matplotlib.pyplot as plt
import json


from dqn_utils import *

with open('./resources/configs/config_highway.json', 'r', encoding='utf-8') as json_file:
    config_highway=json.load(json_file)

if __name__ == "__main__":
    # Create the environment
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    env.unwrapped.configure(config_highway)
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

    N_episodes = 10

    agent = DQN(*arguments)

        
    # Run the training loop
    losses = train(env, agent, N_episodes, eval_every=50)

    plt.plot(losses)
    #plt.show()

    # Evaluate the final policy
    rewards = eval_agent(agent, env, 20)
    print("")
    print("mean reward after training = ", np.mean(rewards))

    # Run the trained model and record video
    #model = DQN.load("highway_dqn/model", env=env)
    env = RecordVideo(
        env, video_folder="models/highway_dqn/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 50})  # Higher FPS for rendering
    state, _ = env.reset()
    for videos in range(5):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action = agent.get_action(state, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state

            # Render
            env.render()
        
    
    env.close()