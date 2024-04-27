import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt
import time

from tools.tools_constants import (
    DICT_CONFIGS
)
from tools.tools_reinforce import (
    run_one_episode,
    eval_agent,
    train,
    Reinforce,
    ReinforceBatch,
    Reinforce
)
from tools.tools_ddpg import (
    DDPGAgent,
    trainer
)

if __name__ == "__main__":
    # Create the environment
    env = gym.make("parking-v0", render_mode="rgb_array")
    env.unwrapped.configure(DICT_CONFIGS["parking"])
    obs, info = env.reset()

    action_space = env.action_space
    observation_space = env.observation_space

    ### DDPG 

    # max_episodes = 100
    # max_steps = 500
    # batch_size = 32

    # gamma = 0.99
    # tau = 1e-2
    # buffer_maxlen = 100000
    # critic_lr = 1e-3
    # actor_lr = 1e-3

    # agent = DDPGAgent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr)
    # episode_rewards = trainer(env, agent, max_episodes, max_steps, batch_size,action_noise=0.1)

    ### REINFORCEBatch

    agent = ReinforceBatch(env.action_space, env.observation_space, gamma=0.99, episode_batch_size=32, learning_rate=0.0001)
    start_time = time.time()

    mean_reward_before = np.mean(eval_agent(agent, env, 200))
    # Run the training loop
    train(env, agent, n_episodes=20000, eval_every=50, reward_threshold=200, n_eval=10)
    end_time = time.time()

    print("mean reward before training = ", mean_reward_before)
    print(f"Training time: {(end_time - start_time)//60} min" )

    # Evaluate the final policy
    print("mean reward after training = ", np.mean(eval_agent(agent, env, 200)))


    env = RecordVideo(
        env, video_folder="models/parking_reinforce/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 50})  # Higher FPS for rendering
    state, _ = env.reset()
    for videos in range(5):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action =  env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state

            done = terminated or truncated or info['crashed'] == True

            # Render
            env.render()
        
    
    env.close()