import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from env.gymnasium_env import ScumEnv
from agent.agent_pool import AgentPool
from config.constants import Constants as C
import pandas as pd

def evaluate(paths, models, agent_number_playing: int=None, verbose=False):
    env = ScumEnv(5)
    agent_pool = AgentPool(5)
    agent_pool = agent_pool.create_agents_with_paths(paths, models)
    if agent_number_playing:
        agent_pool.get_agent(agent_number_playing).set_playing(True)
    winners = []
    for _ in range(1_000):
        _ , winner = env.run_episode(agent_pool, C.DISCOUNT, verbose)
        winners.append(winner)

    df = pd.DataFrame(winners, columns=["winner_player"])
    os.makedirs("./analytics/data/evaluation/", exist_ok=True)
    df.to_parquet("./analytics/data/evaluation/evaluation.parquet")
    return df

if __name__ == '__main__':
    models = ["large_sep_arch", "large_sep_arch", "large_sep_arch", "large_sep_arch", "large"]
    paths = [
        "./learned_models_sep_arch/model_634000.pt",
        "./learned_models_sep_arch/model_634000.pt",
        "./learned_models_sep_arch/model_634000.pt",
        "./learned_models_sep_arch/model_634000.pt",
        None,
        ]
    evaluate(paths, models, None, False)