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

PATH_RESULTS = "results/"
PATH_RESOURCES = "resources/"
PATH_MODELS = "models/"
TRAIN_MODE = False
PATH_CONFIGS = PATH_RESOURCES + "configs/"
DICT_CONFIGS = {
    "highway": load_json_file(PATH_CONFIGS + "config_highway.json"),
    "racetrack": load_json_file(PATH_CONFIGS + "config_racetrack.json"),
    "roundabout": load_json_file(PATH_CONFIGS + "config_roundabout.json")
}
