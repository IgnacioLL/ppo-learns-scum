import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


from agent.a2c_scum import A2CScum

from config.constants import Constants as C

DEFAULT_HYPERPARAMS = {
    "number_of_agents": 5,
    "learning_rate": 5e-5,
    "model_size": "large", 
    "policy_error_coef": 1,
    "value_error_coef": 1, # The advantadge is of low magnitude
    "entropy_coef": 0.1,
    "comment": "-increased-entr-sep-arch"
}


if __name__ == "__main__":
    A2CScum(**DEFAULT_HYPERPARAMS, callback=None).learn(total_episodes=C.EPISODES)