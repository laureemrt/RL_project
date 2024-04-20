import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt

from tools.tools_constants import (
    DICT_CONFIGS
)
from tools.tools_reinforce import (
    run_one_episode,
    eval_agent,
    train,
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

    # agent = RandomAgent(env.observation_space, env.action_space)

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

    ### REINFORCE
    gamma = 0.99
    episode_batch_size = 1
    learning_rate = 1e-2

    agent = Reinforce(
            action_space,
            observation_space,
            gamma,
            episode_batch_size,
            learning_rate,
            )
    N_episodes = 300

    print("mean reward before training = ", np.mean(eval_agent(agent, env, 200)))
    # Run the training loop
    train(env, agent, N_episodes, eval_every=50,)

    # Evaluate the final policy
    print("mean reward after training = ", np.mean(eval_agent(agent, env, 200)))


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