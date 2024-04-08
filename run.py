###############
### Imports ###
###############

### Python imports ###

import matplotlib.pyplot as plt
import gymnasium as gym

### Local imports ###

from tools.tools_constants import (
    DICT_CONFIGS
)

#################
### Main code ###
#################

highway = True
parking = False
racetrack = False

if highway:
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    env.unwrapped.configure(DICT_CONFIGS["highway"])
    env.reset()
    for _ in range(100):
        action =  env.action_space.sample()
        # action = env.action_type.actions_indexes["IDLE"]
        obs, reward, done, truncated, info = env.step(action)
        env.render()

if parking:
    env = gym.make("parking-v0", render_mode="rgb_array")
    env.unwrapped.configure(DICT_CONFIGS["parking"])
    env.reset()
    for _ in range(100):
        action =  env.action_space.sample()
        # action = env.action_type.actions_indexes["IDLE"]
        obs, reward, done, truncated, info = env.step(action)
        env.render()

if racetrack:
    env = gym.make("racetrack-v0", render_mode="rgb_array")
    env.unwrapped.configure(DICT_CONFIGS["racetrack"])
    env.reset()
    for _ in range(100):
        action =  env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()

plt.imshow(env.render())
plt.show()
