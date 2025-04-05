import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agent.agent_pool import AgentPool
from config.constants import Constants as C
from env.gymnasium_env import ScumEnv
from db.db import MongoDBManager
from utils import utils
import pandas as pd

from typing import List, Dict
from tqdm import tqdm

class ScumTournament:
    def __init__(self, mongodb_manager: MongoDBManager):
        self.env = ScumEnv(C.NUMBER_OF_AGENTS)
        self.mongodb_manager = mongodb_manager
        self.leaderboard = self._initialize_leaderboard()

    def _initialize_leaderboard(self):
        model_list = self.mongodb_manager.find_many(C.NAME_COLLECTION_MODELS_PARAMS)
        return { model_params['model_id']: {'model_tag': model_params['model_tag'], 'wins': 0} for model_params in model_list}
    
    def update_leaderboard(self, latest_round_wins: Dict[str, int]):
        for model_id in latest_round_wins.keys():
            self.leaderboard[model_id]['wins'] += latest_round_wins[model_id]

    def play_tournament(self, n_rounds, n_episodes_x_round):
        for _ in tqdm(range(n_rounds), ascii=True, unit=' rounds'):
            self.play_round(n_episodes_x_round)
        return self
    
    def play_round(self, episodes):
        agent_pools = self.make_agent_pools()
        for agent_pool in agent_pools:
            env = ScumEnv(C.NUMBER_OF_AGENTS)
            for _ in range(episodes):
                agent_pool.randomize_order()
                _, wins = env.run_episode(agent_pool, C.DISCOUNT)
                agent_pool.append_win_to_historic_record_to_each_agent(wins)
            wins_per_agent = agent_pool.extract_wins_agents()
            self.update_leaderboard(wins_per_agent)
    
    def make_agent_pools(self) -> List[AgentPool]:
        model_params = self.extract_models_params_latest_checkpoint()
        model_params_groups = utils.divide_into_subgroups(model_params, k=len(self.leaderboard.keys()) // C.NUMBER_OF_AGENTS)

        agent_pools = []
        for group_params in model_params_groups:
            agent_pool = AgentPool(C.NUMBER_OF_AGENTS, self.mongodb_manager)
            agent_pool = agent_pool.create_agents_with_paths(group_params)
            agent_pools.append(agent_pool)

        return agent_pools

    def extract_models_params_latest_checkpoint(self):
        model_list = self.mongodb_manager.find_many(C.NAME_COLLECTION_MODELS_PARAMS)
        
        agent_params = []
        for model_params in model_list:
            query= {'model_id': model_params['model_id'], 'current_episode': model_params['current_episode']}
            params = self.mongodb_manager.find_one(C.NAME_COLLECTION_CHECKPOINTS, query)
            params.pop('_id')
            agent_params.append(params)

        return agent_params

    def get_leaderboard(self):
        df = pd.DataFrame.from_records(self.leaderboard).T
        df = df.sort_values('wins', ascending=False)
        return df

if __name__ == '__main__':
    mongodb = MongoDBManager(database="population-based-training")
    scum_tournament = ScumTournament(mongodb)
    scum_tournament = scum_tournament.play_tournament(50, 10)
    print(scum_tournament.get_leaderboard())
