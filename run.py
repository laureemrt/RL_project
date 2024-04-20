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
racetrack = False
roundabout = False

if highway:
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure(DICT_CONFIGS["highway"])
    env.reset()
    for _ in range(100):
        action =  env.action_space.sample()
        print(action)
        # action = env.action_type.actions_indexes["IDLE"]
        obs, reward, done, truncated, info = env.step(action)
        env.render()

if racetrack:
    env = gym.make("racetrack-v0", render_mode="rgb_array")
    env.unwrapped.configure(DICT_CONFIGS["racetrack"])
    env.reset()
    for _ in range(100):
        action =  env.action_space.sample()
        print(action)
        obs, reward, done, truncated, info = env.step(action)
        env.render()

if roundabout:
    env = gym.make("roundabout-v0", render_mode="rgb_array")
    env.unwrapped.configure(DICT_CONFIGS["roundabout"])
    env.reset()
    for _ in range(100):
        action =  env.action_space.sample()
        # action = env.action_type.actions_indexes["IDLE"]
        obs, reward, done, truncated, info = env.step(action)
        env.render()

plt.imshow(env.render())
plt.show()
