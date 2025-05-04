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

from scum_model.agent.agent_training import AgentTraining

from config.constants import Constants as C


N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
EVAL_FREQ = C.EPISODES // 10
N_EVAL_EPISODES = 200

DEFAULT_HYPERPARAMS = {
    "number_of_agents": 5,
    "load_checkpoints": False,
}


def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for A2C hyperparameters."""
    learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    entropy_coef = trial.suggest_float("entropy_coef", 1e-4, 1e-1, log=True)
    model = trial.suggest_categorical("model", ["small", "medium", "big", "large"])

    return {
        "learning_rate": learning_rate,
        "model": model,
        "entropy_coef": entropy_coef
    }

class TrialEvalCallback(BaseCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
    ):
        super().__init__()

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.last_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial, episodes=C.EPISODES) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_a2c_params(trial))
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ)
    nan_encountered = False

    print("Learning rate is: ", kwargs['learning_rate'])
    print("Model is: ", kwargs['model'])
    print("Entropy coefficient is: ", kwargs['entropy_coef'])
    
    try:
        AgentTraining(**kwargs, callback=eval_callback).learn(total_episodes=episodes)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        pass

    print("Number of done trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))