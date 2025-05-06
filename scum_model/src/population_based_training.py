import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import uuid

from scipy.stats import loguniform
from agent.agent_training import AgentTraining
from agent.agent_pool import AgentPool
from config.constants import Constants as C
from db.db import MongoDBManager
from evaluation.tournament import ScumTournament
from utils import db_utils

from typing import Dict, Any, List

class PopulationBasedTraining:
    def __init__(
            self,
            n_iter_total=C.N_ITER_TOTAL,
            n_iter_against_another_model=C.N_ITER_AGAINST_ANOTHER_MODEL,
            pruning_rounds=C.PRUNING_ROUNDS,
            number_of_models_in_parallel=C.NUMBER_OF_MODELS_IN_PARALLEL,
            models_training: List[str]=None
            ):
        
        self.n_iter_total = n_iter_total
        self.n_iter_against_another_model = n_iter_against_another_model
        self.number_of_models_in_parallel = number_of_models_in_parallel
        self.total_round = self.n_iter_total // self.n_iter_against_another_model
        self.pruning_rounds = pruning_rounds
        self.mongodb_manager = MongoDBManager(database="population-based-training")
        self.models_training = db_utils.extract_all_models_in_db(self.mongodb_manager) if models_training is None else models_training


    def initialize_training_with_reinitialization(self, number_of_new_models: int=C.NUMBER_OF_MODELS_IN_PARALLEL):
        self._drop_collections_training()
        self.initialize_training(number_of_new_models)

    def _drop_collections_training(self):
        self.mongodb_manager.drop_collection(C.NAME_COLLECTION_CHECKPOINTS)
        self.mongodb_manager.drop_collection(C.NAME_COLLECTION_MODELS_PARAMS)

    def initialize_training(self, number_of_new_models: int=C.NUMBER_OF_MODELS_IN_PARALLEL):
        list_of_params = [self._generate_params() for _ in range(number_of_new_models)]
        self.mongodb_manager.insert_many(C.NAME_COLLECTION_MODELS_PARAMS, list_of_params)
        
        data =  [
            {
                'model_id': params['model_id'], 
                'load_model_path': params['load_model_path'], 
                'current_episode': 0, 
                'model_size': params["model_size"], 
                'model_tag': params['model_tag']
            }
                for params in list_of_params
        ]
        
        self.mongodb_manager.insert_many(C.NAME_COLLECTION_CHECKPOINTS, data)
        self.models_training = self.models_training + [params['model_id'] for params in list_of_params]


    def _generate_params(self):
        params = {
            "learning_rate": float(loguniform.rvs(1e-5, 1e-3)),
            "model_size": np.random.choice([
                'large-sep-arch', 'big-sep-arch', 'medium-sep-arch', 'small-sep-arch', 
                'large', 'big', 'medium', 'small']), 
            "policy_error_coef": float(loguniform.rvs(0.1, 10)),
            "value_error_coef": 1, # The advantadge is of low magnitude
            "entropy_coef": float(loguniform.rvs(1e-7, 1.5)),
            "current_episode": 0,
            'load_model_path': None,
            'clip': float(loguniform.rvs(0.05, 0.5)),
            'epochs': int(np.random.randint(1, 25)),
            'batch_size': int(np.random.choice([8, 16, 32, 64, 128, 512], p= [0.1, 0.1, 0.3, 0.3, 0.1, 0.1])),
            'discount': float(loguniform.rvs(0.95, 0.99999)),
            'pruned': False,
        }
        params['model_id'] = str(uuid.uuid4())
        params['model_tag'] = str(round(params['learning_rate'], 5)) + "-" + params['model_size'] + '-' + str(round(params['policy_error_coef'],2)) + '-' + str(round(params['value_error_coef'], 2)) + '-' + str(round(params['entropy_coef'], 2))
        return params


    def training(self, training_against_different_models: bool=False):
        current_round = self.determine_current_round()
        for round in range(current_round, self.total_round):
            begin_episode = self.n_iter_against_another_model*round
            end_episode = self.n_iter_against_another_model*(round+1)
            
            number_of_models_in_current_round = self.find_number_agents_in_round()
            if number_of_models_in_current_round == self.number_of_models_in_parallel and round in self.pruning_rounds:
                print(f"Pruning round {round}")
                self.prune()
            print(f"Number of models still in round {number_of_models_in_current_round}")
            for _ in range(number_of_models_in_current_round):
                model_params = self.find_least_trained_agent()
                if training_against_different_models:
                    rivals_model_params = self.find_worth_table(model_params)
                    parameters = [model_params] + rivals_model_params
                else:
                    rival_model_params = self.find_worth_opponent(model_params)
                    parameters = [model_params, rival_model_params, rival_model_params, rival_model_params, rival_model_params]

                agent_pool = AgentPool(5, self.mongodb_manager)
                agent_pool = agent_pool.create_agents_with_parameters(parameters)

                agent_trainer = AgentTraining(5, agent_pool, self.n_iter_against_another_model)
                agent_trainer.learn(begin_episode, end_episode)
                self._update_checkpoint(model_params, end_episode)

    def prune(self) -> None:
        tournament = ScumTournament(self.mongodb_manager)
        tournament.play_tournament(50, 10)
        leaderboard = tournament.get_leaderboard()
        models_to_prune_ids = leaderboard.tail(5)['model_id'].to_list()

        prune_query = {'model_id': {'$in': models_to_prune_ids}}
        update_data = {'$set': {'pruned': True}}
        self.mongodb_manager.update_many(
            collection=C.NAME_COLLECTION_MODELS_PARAMS,
            query=prune_query,
            update=update_data
        )

        
    
    def determine_current_round(self):
        return self.find_least_trained_agent()['current_episode'] // self.n_iter_against_another_model

    def _update_checkpoint(self, model_params, end_episode):
        path = f"{C.MODELS_PATH}model_{model_params['model_id']}_{str(end_episode)}.pt"
        self.mongodb_manager.update_one(
            collection=C.NAME_COLLECTION_MODELS_PARAMS,  
            query={'model_id': model_params['model_id']}, 
            update={'$set': 
                    {'current_episode': end_episode, 
                        'load_model_path': path}
                        }
        )


    def find_least_trained_agent(self) -> Dict[str, Any]:
        list_of_params = self.mongodb_manager.find_many(C.NAME_COLLECTION_MODELS_PARAMS, query={'pruned': False})
        list_of_params = [model_params for model_params in list_of_params if model_params['model_id'] in self.models_training]
        min_episode = min([params["current_episode"] for params in list_of_params])
        model_params = [params for params in list_of_params if params["current_episode"] == min_episode]
        model_param: Dict = np.random.choice(model_params)
        model_param.pop('_id')
        model_param.pop('pruned')
        model_param['training'] = True
        return model_param
    
    def find_number_agents_in_round(self) -> Dict[str, Any]:
        list_of_params = self.mongodb_manager.find_many(C.NAME_COLLECTION_MODELS_PARAMS)
        list_of_params = [model_params for model_params in list_of_params if model_params['model_id'] in self.models_training]

        min_episode = min([params["current_episode"] for params in list_of_params])
        model_params = [params for params in list_of_params if params["current_episode"] == min_episode]
        return len(model_params)
    
    def find_worth_opponent(self, training_model) -> Dict[str, Any]:
        "The objective is to find a no-so good model."
        checkpoints = self.mongodb_manager.find_many(C.NAME_COLLECTION_CHECKPOINTS)
        candidate_checkpoints = [
            model_checkpoint
            for model_checkpoint in checkpoints
            if (model_checkpoint["current_episode"] <= training_model['current_episode']) &
              (model_checkpoint["current_episode"] >= (training_model['current_episode'] // 1.5))
        ]
        chosen_params: Dict = np.random.choice(candidate_checkpoints)
        chosen_params.pop('_id')
        return chosen_params
    
    def find_worth_table(self, training_model) -> List[Dict[str, Any]]:
        "The objective is to find a no-so good model."
        checkpoints = self.mongodb_manager.find_many(C.NAME_COLLECTION_CHECKPOINTS)
        candidate_checkpoints = [
            model_checkpoint
            for model_checkpoint in checkpoints
            if (model_checkpoint["current_episode"] <= training_model['current_episode']) &
              (model_checkpoint["current_episode"] >= (training_model['current_episode'] // 1.5)) & 
              (model_checkpoint['model_id'] in self.models_training)
        ]
        chosen_params: List[Dict] = list(np.random.choice(candidate_checkpoints, size=4, replace=False))
        [x.pop('_id') for x in chosen_params]
        print(type(chosen_params))
        return chosen_params
        
        

if __name__ == '__main__':
    # models_id = [
    #     "fb50b8c8-3848-4b5e-a144-5efb6a256dad",
    #     "5e09f893-fb82-463e-8eea-4c9f5b429448",
    #     "f8400821-9a91-49e3-b5e4-69d7ab0791ff",
    #     "631941e3-f3db-438e-9ca4-12605fd0423c",
    #     "c401ab2d-0403-4468-af57-aea518cf56cc"
    # ]
    pbt = PopulationBasedTraining(600_000, 5_000, 30)
    pbt.initialize_training_with_reinitialization(30)
    pbt.training(True)
        