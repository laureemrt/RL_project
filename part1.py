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
    buffer_capacity = 15_000
    update_target_every = 32

    epsilon_start = 1
    epsilon_decay = 0.99975
    epsilon_min = 0.001

    learning_rate = 5e-4

    arguments = (action_space,
            observation_space,
            gamma,
            batch_size,
            buffer_capacity,
            update_target_every, 
            epsilon_start, 
            epsilon_decay, 
            epsilon_min,
            learning_rate,
        )

    N_episodes = 5_000

    agent = DQN(*arguments)

        
    # Run the training loop
    losses, speed_avg = train(env, agent, N_episodes, eval_every=100)

    plt.plot(losses)
    plt.show()

    plt.plot(speed_avg)
    plt.show()

    # Run the trained model and record video
    env = RecordVideo(
        env, video_folder="models/highway_dqn/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 50})  # Higher FPS for rendering
    for videos in range(5):
        done = truncated = False
        state, info = env.reset()
        time_step_reward = 0.01
        total_reward = 0
        while not (done or truncated):
            state_tensor = torch.tensor([state], dtype=torch.float32).to(agent.device)
            action = agent.get_action(state_tensor).item()
            next_state, reward, done, truncated, info = env.step(action)

            next_state = next_state.flatten()

            reward, on_road_reward, speed_reward, collision_reward = agent.calculate_reward(env, info, reward)
            reward += agent.n_steps * time_step_reward
            total_reward += reward
            terminated = done or truncated
            agent.buffer.push(state, action, reward, next_state, terminated)

            if on_road_reward > 0:
                agent.update(terminated)
                state = next_state
                
            if on_road_reward <= 0 or speed_reward <= 0 or collision_reward < 0:
                break
            if done or truncated:
                break

            # Render
            env.render()

    env.close()