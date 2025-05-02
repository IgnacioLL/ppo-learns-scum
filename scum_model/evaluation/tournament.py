import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agent.agent_pool import AgentPool
from config.constants import Constants as C
from env.gymnasium_env import ScumEnv
from db.db import MongoDBManager
from utils import utils, db_utils
import pandas as pd

from typing import List, Dict, Any
from tqdm import tqdm
import torch


class ScumTournament:
    def __init__(self, mongodb_manager: MongoDBManager, models: List[str]=None, runs=None):
        self.env = ScumEnv(C.NUMBER_OF_AGENTS)
        self.mongodb_manager = mongodb_manager
        self.models = db_utils.extract_all_models_in_db(mongodb_manager) if models is None else models
        if len(models) != 1:
            self.leaderboard = self._initialize_leaderboard_different_models()
        else:
            self.leaderboard = self._initialize_leaderboard_same_model(models[0], runs)

    def _initialize_leaderboard_same_model(self, model_id, runs: List[int]): 
        model_list = self.mongodb_manager.find_many(C.NAME_COLLECTION_CHECKPOINTS, query={'model_id': model_id, 'current_episode': {"$in": runs}})
        return { 
            model_params['model_id'] + "-" + str(model_params['current_episode']): {'model_tag': model_params['model_tag'], 'wins': 0} 
            for model_params in model_list
            }

    def _initialize_leaderboard_different_models(self):
        model_list = self.mongodb_manager.find_many(C.NAME_COLLECTION_MODELS_PARAMS, query={'model_id': {"$in": self.models}})
        return { model_params['model_id']: {'model_tag': model_params['model_tag'], 'wins': 0} for model_params in model_list}
    
    def update_leaderboard(self, latest_round_wins: Dict[str, int]):
        for model_id in latest_round_wins.keys():
            self.leaderboard[model_id]['wins'] += latest_round_wins[model_id]

    def play_tournament(self, n_rounds, n_episodes_x_round):
        for _ in tqdm(range(n_rounds), ascii=True, unit=' rounds'):
            self.play_round(n_episodes_x_round)

    def play_tournament_against_own_model(self, n_rounds, n_episodes_x_round, **same_model_kwargs):
        for _ in tqdm(range(n_rounds), ascii=True, unit=' rounds'):
            self.play_round(n_episodes_x_round, True, **same_model_kwargs)    

    def play_round(self, episodes, same_model=False, **same_model_kwargs):
        utils.log_vram_memory("Before make agent pools")
        agent_pools = self.make_agent_pools(same_model, **same_model_kwargs)
        utils.log_vram_memory("After making agent pools")
        for agent_pool in agent_pools:
            env = ScumEnv(C.NUMBER_OF_AGENTS)
            for _ in range(episodes):
                agent_pool.randomize_order()
                with torch.no_grad():
                    _, wins = env.run_episode(agent_pool, C.DISCOUNT, save_in_buffer=False)

                agent_pool.append_win_to_historic_record_to_each_agent(wins)
            wins_per_agent = agent_pool.extract_wins_agents_w_current_episode() if same_model else agent_pool.extract_wins_agents()
            self.update_leaderboard(wins_per_agent)
                
        for agent_pool in agent_pools:
            del agent_pool
        utils.log_vram_memory("After deletion of agent pools")


    def make_agent_pools(self, same_model=False, **same_model_kwargs) -> List[AgentPool]:
        model_params = self.extract_models_params_from_single_model(**same_model_kwargs) if same_model else self.extract_models_params_latest_checkpoint()
        model_params_groups = utils.divide_into_subgroups(model_params, k=len(self.leaderboard.keys()) // C.NUMBER_OF_AGENTS)
        agent_pools = []
        for group_params in model_params_groups:
            agent_pool = AgentPool(C.NUMBER_OF_AGENTS, self.mongodb_manager)
            agent_pool = agent_pool.create_agents_with_paths(group_params)
            agent_pools.append(agent_pool)
        return agent_pools

    def extract_models_params_latest_checkpoint(self):
        model_list = self.mongodb_manager.find_many(C.NAME_COLLECTION_MODELS_PARAMS, query= {'model_id': {'$in': self.models}})
        
        agent_params = []
        for model_params in model_list:
            query= {'model_id': model_params['model_id'], 'current_episode': model_params['current_episode']}
            params = self.mongodb_manager.find_one(C.NAME_COLLECTION_CHECKPOINTS, query)
            params.pop('_id')
            agent_params.append(params)

        return agent_params
    
    def extract_models_params_from_single_model(self, model_id: str, runs: List[int]) -> Dict[str, Any]:
        final_models_params_list = []
        for run in runs:
            query= {'model_id': model_id, 'current_episode': run}
            params = self.mongodb_manager.find_one(C.NAME_COLLECTION_CHECKPOINTS, query)
            params.pop('_id')
            final_models_params_list.append(params)
        return final_models_params_list

    def get_leaderboard(self):
        df = pd.DataFrame.from_records(self.leaderboard).T
        df = df.sort_values('wins', ascending=False)
        return df

if __name__ == '__main__':
    mongodb = MongoDBManager(database="population-based-training")
    scum_tournament = ScumTournament(mongodb)
    scum_tournament = scum_tournament.play_tournament(50, 10)
    print(scum_tournament.get_leaderboard())
