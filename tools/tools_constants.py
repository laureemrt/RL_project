"""
Tools module containing the main constants of the code.
"""

###############
### Imports ###
###############

### Local imports ###

from tools.tools_basis import (
    load_json_file
)

#################
### Constants ###
#################

### Paths ###

PATH_RESULTS = "results/"
PATH_RESOURCES = "resources/"
PATH_MODELS = "models/"

### Configs ###

PATH_CONFIGS = PATH_RESOURCES + "configs/"
DICT_CONFIGS = {
    "highway": load_json_file(PATH_CONFIGS + "config_highway.json"),
    "parking": load_json_file(PATH_CONFIGS + "config_parking.json"),
    "racetrack": load_json_file(PATH_CONFIGS + "config_racetrack.json")
}

### Training ###

TRAIN_MODE = True
NUMBER_VIDEOS_TO_GENERATE = 10
