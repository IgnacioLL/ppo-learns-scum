""" Optuna example that optimizes the hyperparameters of
a reinforcement learning agent using A2C implementation from Stable-Baselines3
on a Gymnasium environment.

This is a simplified version of what can be found in https://github.com/DLR-RM/rl-baselines3-zoo.

You can run this example as follows:
    $ python sb3_simple.py

"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


from typing import Any
from typing import Dict

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from stable_baselines3.common.callbacks import BaseCallback

import torch
import numpy as np

from agent.a2c_scum import A2CScum

from config.constants import Constants as C


N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
EVAL_FREQ = C.EPISODES // 10
N_EVAL_EPISODES = 200

DEFAULT_HYPERPARAMS = {
    "number_of_agents": 5,
    "load_checkpoints": False,
    "learning_rate": 1e-3, 
    "model":"large", 
    "policy_error_coef": 1,
    "value_error_coef": 0.1,
    "entropy_coef": 0
}


if __name__ == "__main__":
    A2CScum(**DEFAULT_HYPERPARAMS, callback=None).learn(total_episodes=C.EPISODES)