"""
Main python script for Part 3 of the project.
Implement PPO model from StableBaselines for the racetrack configuration.
"""

###############
### Imports ###
###############

### Python imports ###

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

### Local imports ###

from tools.tools_constants import (
    DICT_CONFIGS,
    PATH_MODELS,
    PATH_RESULTS,
    TRAIN_MODE,
    NUMBER_VIDEOS_TO_GENERATE
)

#################
### Main code ###
#################

if __name__ == "__main__":
    nb_cpu = 6
    batch_size = 64
    env = make_vec_env("racetrack-v0", n_envs=nb_cpu, vec_env_cls=SubprocVecEnv)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=batch_size * 12 // nb_cpu,
        batch_size=batch_size,
        n_epochs=5, # 10
        learning_rate=5e-4,
        gamma=0.9,
        verbose=2,
        seed=123
    )
    # Train the model
    if TRAIN_MODE:
        model.learn(total_timesteps=int(1e5))
        model.save(PATH_MODELS + "racetrack_ppo/model")
        del model

    # Run the algorithm
    model = PPO.load(PATH_MODELS + "racetrack_ppo/model", env=env)

    env = gym.make("racetrack-v0", render_mode="rgb_array")
    env.unwrapped.configure(DICT_CONFIGS["racetrack"])
    obs, info = env.reset()

    env = RecordVideo(
        env,
        video_folder=PATH_RESULTS + "racetrack_ppo/",
        episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 50})

    for video in range(NUMBER_VIDEOS_TO_GENERATE):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()
