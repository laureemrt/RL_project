import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt

from tools.tools_constants import (
    DICT_CONFIGS
)
from tools.tools_part2 import (
    run_one_episode,
    eval_agent,
    RandomAgent,
    ReinforceSkeleton
)


if __name__ == "__main__":
    # Create the environment
    env = gym.make("parking-v0", render_mode="rgb_array")
    env.unwrapped.configure(DICT_CONFIGS["parking"])
    obs, info = env.reset()

    # agent = RandomAgent(env.observation_space, env.action_space)


    action_space = env.action_space
    print(action_space.shape[0])
    observation_space = env.observation_space
    gamma = 0.99
    episode_batch_size = 1
    learning_rate = 1e-2

    agent = ReinforceSkeleton(action_space,
            observation_space,
            gamma,
            episode_batch_size,
            learning_rate,)
    
    run_one_episode(env, agent, display=False)


    # run_one_episode(env, agent, display=True)

    # print(f'Average over 5 runs : {np.mean(eval_agent(agent, env))}')

    # Run the trained model and record video
    #model = DQN.load("parking_dqn/model", env=env)
    # env = RecordVideo(
    #     env, video_folder="models/parking_dqn/videos", episode_trigger=lambda e: True
    # )
    # env.unwrapped.set_record_video_wrapper(env)
    # env.configure({"simulation_frequency": 50})  # Higher FPS for rendering
    # state, _ = env.reset()
    # for videos in range(5):
    #     done = truncated = False
    #     obs, info = env.reset()
    #     while not (done or truncated):
    #         action = agent.get_action(state, env)
    #         next_state, reward, terminated, truncated, _ = env.step(action)
    #         state = next_state

    #         # Render
    #         env.render()
        
    
    # env.close()