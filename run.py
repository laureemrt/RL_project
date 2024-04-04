from resources.config import (
    config_highway,
    config_racetrack,
    config_roundabout
)
import matplotlib.pyplot as plt
import gymnasium as gym


highway = False
racetrack = True
roundabout = False

if highway:
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure(config_highway)
    env.reset()
    for _ in range(100):
        action = env.action_type.actions_indexes["IDLE"]
        obs, reward, done, truncated, info = env.step(action)
        env.render()

if racetrack:
    env = gym.make("racetrack-v0", render_mode="rgb_array")
    env.unwrapped.configure(config_racetrack)
    env.reset()
    for _ in range(100):
        action =  env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()

if roundabout:
    env = gym.make("roundabout-v0", render_mode="rgb_array")
    env.unwrapped.configure(config_roundabout)
    env.reset()
    for _ in range(100):
        # action =  env.action_space.sample()
        action = env.action_type.actions_indexes["IDLE"]
        obs, reward, done, truncated, info = env.step(action)
        env.render()


plt.imshow(env.render())
plt.show()

