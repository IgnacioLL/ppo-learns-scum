import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from environment.gymnasium_env import ScumEnv
from agent.agent_pool import AgentPool
from config.constants import Constants as C
import pandas as pd

from db.db import MongoDBManager

def evaluate(parameters, verbose=False):
    env = ScumEnv(5)
    agent_pool = AgentPool(5)
    agent_pool = agent_pool.create_agents_with_parameters(parameters)
    winners = []
    for _ in range(10):
        agent_pool = agent_pool.randomize_order()
        _ , winner = env.run_episode(agent_pool, C.DISCOUNT, verbose)
    agent_ids = [agent_pool.get_agent(i).model_id + "-" + str(i) for i in range(agent_pool.number_of_agents)]
    df = pd.DataFrame(winners, columns=agent_ids)
    os.makedirs("./analytics/data/evaluation/", exist_ok=True)
    df.to_parquet("./analytics/data/evaluation/evaluation.parquet")
    print(df)
    return df



if __name__ == '__main__':
    mongodb_manager = MongoDBManager(database="population-based-training")
    model_id = "fb50b8c8-3848-4b5e-a144-5efb6a256dad"
    run = 60_000
    model_param = mongodb_manager.find_one(C.NAME_COLLECTION_CHECKPOINTS, query={'model_id': model_id, 'current_episode': run})
    model_param.pop('_id')

    parameters = [
        {'model_id': 'heuristic', 'model_size': 'heuristic', 'model_tag': 'heuristic'},
        {'model_id': 'heuristic', 'model_size': 'heuristic', 'model_tag': 'heuristic'},
        {'model_id': 'heuristic', 'model_size': 'heuristic', 'model_tag': 'heuristic'},
        {'model_id': 'heuristic', 'model_size': 'heuristic', 'model_tag': 'heuristic'},
        model_param
        ]
    evaluate(parameters)