# RL Project

## Table of contents

- [RL Project](#rl-project)
  - [Table of contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
    - [Cloning the repository](#cloning-the-repository)
    - [Creation of a virtual environment](#creation-of-a-virtual-environment)
    - [Installation of the necessary librairies](#installation-of-the-necessary-librairies)
    - [Modifications made in the `highway-env` library](#modifications-made-in-the-highway-env-library)
  - [Utilization](#utilization)
    - [Part 1](#part-1)
    - [Part 2](#part-2)
    - [Part 3](#part-3)
  - [Architecture of the project](#architecture-of-the-project)

## Introduction

This project has been realized in the scope of the RL course at CentraleSupélec by:
- Gwénaëlle BIENVENUE
- Laure-Emilie MARTIN
- Agathe PLU

## Installation

### Cloning the repository

To clone the github repository, you have to search the clone button on the main page of the project. Then click on it and select `https` or `ssh` depending on your favorite mode of connexion. Copy the given id and then open a terminal on your computer, go to the folder where you want to install the project and use the following command:

```bash
git clone <your copied content>
```

### Creation of a virtual environment

You might want to use a virtual environment to execute the code. To do so, use the following command:

```bash
python -m virtualenv venv
```

To start it, use the command on *Windows*:

```bash
venv/Scripts/Activate.ps1
```
git pu
Or for *MacOS* and *Linux*:

```bash
venv/Scripts/activate
```

### Installation of the necessary librairies

To execute this software, you need several *Python* librairies, specified in the `requirements.txt` file. To install them, use the following command:

```bash
pip install -r requirements.txt
```

### Modifications made in the `highway-env` library

We modified a few things in the `highway-env` library, because of bugs which have not been resolved in the latest release of the library.

First of all, in the file `racetrack_env.py` which can be found at the directory `venv\lib\site-packages\highway_env\envs\`, we modified the last function creating the vehicles by:

```python
  def _make_vehicles(self) -> None:
      """
      Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
      """
      rng = self.np_random

      # Controlled vehicles
      self.controlled_vehicles = []
      for i in range(self.config["controlled_vehicles"]):
          lane_index = ("a", "b", rng.integers(2)) if i == 0 else \
              self.road.network.random_lane_index(rng)
          controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=None,
            longitudinal=rng.uniform(20, 50))

          self.controlled_vehicles.append(controlled_vehicle)
          self.road.vehicles.append(controlled_vehicle)

      # Front vehicle
      vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", lane_index[-1]),
                                        longitudinal=rng.uniform(
                                            low=0,
                                            high=self.road.network.get_lane(("b", "c", 0)).length
                                        ),
                                        speed=6+rng.uniform(high=3))
      self.road.vehicles.append(vehicle)

      # Other vehicles
      list_lines_taken = []
      counter_vehicles = 0
      while counter_vehicles < self.config["other_vehicles"]-1:
          random_lane_index = self.road.network.random_lane_index(rng)
          while random_lane_index in list_lines_taken:
              random_lane_index = self.road.network.random_lane_index(rng)
          vehicle = IDMVehicle.make_on_lane(
              self.road, random_lane_index,
              longitudinal=rng.uniform(
                  low=0,
                  high=self.road.network.get_lane(random_lane_index).length),
              speed=6+rng.uniform(high=3))
          # Prevent early collisions
          for v in self.road.vehicles:
              if np.linalg.norm(vehicle.position - v.position) < 20:
                  break
          else:
              self.road.vehicles.append(vehicle)
              counter_vehicles += 1
              list_lines_taken.append(random_lane_index)
```
Indeed, the former function was not creating the correct number of vehicles we were specifying in the `config` file, so we changed the source code.

Secondly, there was also an error sometimes while running the racetrack configuration, without any change in the code. We found another bug in the source code, in the file `venv\lib\site-packages\highway_env\road\road.py` in the function `random_lane_index` line 254. We changed the line `_id = np_random.randint(len(self.graph[_from][_to]))` to `_id = np_random.integers(len(self.graph[_from][_to]))`.

## Utilization

### Part 1

### Part 2

Part 2 of the project can be launched by running the following command:
```bash
python part2_parking_env.py`
```

### Part 3

Part 3 of the project can be launched by running the following command:
```bash
python stable_baselines_ppo_racetrack.py`
```

Some configuration can be done before running the code:
- in the file `tools/tools_constants.py`, the constant `TRAIN_MODE` may be set to True ou False, depending if you want to run a new training of the model. For the first time, it must be set to True, because you don't have the model weights saved in the directory `models`.
- in the file `tools/tools_constants.py`, the constant `NUMBER_VIDEOS_TO_GENERATE` may be changed to change the number of videos you want to generate and save in the folder `results` after running the code.

## Architecture of the project

This repository is composed of the following folders:
- `models`, storing the weights of the models trained for each part.
- `resources`, containing the subfolder `configs` to store the different configuration json files.
- `results`, containing the videos while executing one of the three parts.
- `tools`, containing the following Python tools modules:
  - `tools_basis.py`, defining basic tools functions.
  - `tools_constants.py`, defining the main constants of the code, used for configuration.

This repository also contains the following files:
- `main.py`
- `requirements.txt`
- `stable_baselines_ppo_racetrack.py`, main Python file for Part 3.

