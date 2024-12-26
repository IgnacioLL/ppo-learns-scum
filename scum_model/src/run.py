import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


from agent.a2c_scum import A2CScum

from config.constants import Constants as C

DEFAULT_HYPERPARAMS = {
    "number_of_agents": 5,
    "load_checkpoints": False,
    "learning_rate": 1e-3, 
    "model":"large", 
    "policy_error_coef": 1,
    "value_error_coef": 0.01, # The advantadge is of low magnitude
    "entropy_coef": 0
}


if __name__ == "__main__":
    A2CScum(**DEFAULT_HYPERPARAMS, callback=None).learn(total_episodes=C.EPISODES)