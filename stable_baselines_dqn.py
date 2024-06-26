import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

from tools.tools_constants import (
    PATH_MODELS,
    PATH_RESULTS,
    TRAIN_MODE,
    DICT_CONFIGS
)

if __name__ == "__main__":
    # Create the environment
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    env.unwrapped.configure(DICT_CONFIGS["highway"])
    obs, info = env.reset()

    # Create the model
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log=PATH_MODELS + "highway_dqn_2/",
    )

    # Train the model
    if TRAIN_MODE:
        model.learn(total_timesteps=int(2e4))
        model.save(PATH_MODELS + "highway_dqn_2/model")
        del model

    # Run the trained model and record video
    model = DQN.load(PATH_MODELS + "highway_dqn_2/model", env=env)
    env = RecordVideo(
        env, video_folder=PATH_RESULTS + "highway_dqn_2/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 50})  # Higher FPS for rendering

    for videos in range(5):
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
