import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from environment.gymnasium_env import ScumEnv
from agent.agent_pool import AgentPool
from config.constants import Constants as C
import pandas as pd

from tqdm import tqdm
from db.db import MongoDBManager

def evaluate(rounds, parameters, verbose=False):
    env = ScumEnv(5)
    agent_pool = AgentPool(5)
    agent_pool = agent_pool.create_agents_with_parameters(parameters)
    for _ in tqdm(range(rounds), ascii=True, unit=' round/s'):
        agent_pool = agent_pool.randomize_order()
        _ , winner = env.run_episode(agent_pool, C.DISCOUNT, verbose, save_in_buffer=False)
        agent_pool.append_win_to_historic_record_to_each_agent(winner)
        leaderboard = agent_pool.extract_wins_agents_with_invented_id()
    df = pd.DataFrame(leaderboard.values(), index=leaderboard.keys()).T
    os.makedirs("./analytics/data/evaluation/", exist_ok=True)
    df.to_parquet("./analytics/data/evaluation/evaluation.parquet")
    print(df)
    return df

if __name__ == '__main__':
    mongodb_manager = MongoDBManager(database="population-based-training")
    model_id = "4efbec94-219a-4f7b-8805-8be90487019f"
    run = 200_000
    model_param = mongodb_manager.find_one(C.NAME_COLLECTION_CHECKPOINTS, query={'model_id': model_id, 'current_episode': run})
    model_param.pop('_id')

    parameters = [
        {'model_id': 'heuristic', 'model_size': 'heuristic', 'model_tag': 'heuristic'},
        {'model_id': 'heuristic', 'model_size': 'heuristic', 'model_tag': 'heuristic'},
        {'model_id': 'heuristic', 'model_size': 'heuristic', 'model_tag': 'heuristic'},
        {'model_id': 'heuristic', 'model_size': 'heuristic', 'model_tag': 'heuristic'},
        model_param
        ]
    evaluate(1_000, parameters)