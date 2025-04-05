import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


from agent.a2c_scum import A2CScum
from agent.agent_pool import AgentPool
from config.constants import Constants as C

DEFAULT_HYPERPARAMS = {
    "learning_rate": 5e-5,
    "model_size": 'large-sep-arch', 
    "policy_error_coef": 1,
    "value_error_coef": 1, # The advantadge is of low magnitude
    "entropy_coef": 0.1,
    "model_tag": "testing",
    'model_id': 'training'
}


if __name__ == "__main__":
    agent_pool = AgentPool(5, None)
    agent_pool = agent_pool.create_agents_with_nnet_initialization(**DEFAULT_HYPERPARAMS)
    A2CScum(C.NUMBER_OF_AGENTS, agent_pool).learn(0, C.EPISODES)