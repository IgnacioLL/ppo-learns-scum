import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np

from scipy.stats import loguniform
from agent.a2c_scum import A2CScum
from agent.agent_pool import AgentPool
from db.db import MongoDBManager
from config.constants import Constants as C
import uuid

from typing import Dict, Any

class PopulationBasedTraining:
    def __init__(
            self,
            n_iter_total=C.N_ITER_TOTAL,
            n_iter_against_another_model=C.N_ITER_AGAINST_ANOTHER_MODEL,
            number_of_models_in_parallel=C.NUMBER_OF_MODELS_IN_PARALLEL
            ):
        
        self.n_iter_total = n_iter_total
        self.n_iter_against_another_model = n_iter_against_another_model
        self.number_of_models_in_parallel = number_of_models_in_parallel
        self.mongodb_manager = MongoDBManager(database="population-based-training")


    def generate_params(self):
        params = {
            "learning_rate": loguniform.rvs(5e-6, 5e-4),
            "model_size": np.random.choice(['small', 'medium', 'big', 'large', 'small-sep-arch', 'medium-sep-arch', 'big-sep-arch', 'large-sep-arch']), 
            "policy_error_coef": np.random.uniform(0.25, 1.25),
            "value_error_coef": np.random.uniform(0.25, 1.25), # The advantadge is of low magnitude
            "entropy_coef": loguniform.rvs(1e-4, 0.5),
            "current_episode": 0,
            'load_model_path': None, 
        }
        params['model_id'] = str(uuid.uuid4())
        params['model_tag'] = str(round(params['learning_rate'], 5)) + "-" + params['model_size'] + '-' + str(round(params['policy_error_coef'],2)) + '-' + str(round(params['value_error_coef'], 2)) + '-' + str(round(params['entropy_coef'], 2))
        return params

    def initialize_training(self):
        self.mongodb_manager.drop_collection(C.NAME_COLLECTION_CHECKPOINTS)
        self.mongodb_manager.drop_collection(C.NAME_COLLECTION_MODELS_PARAMS)

        list_of_params = [self.generate_params() for _ in range(self.number_of_models_in_parallel)]
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

    def training(self):
        current_round = self.find_least_trained_agent()['current_episode'] // self.n_iter_against_another_model
        for round in range(current_round, self.n_iter_total // self.n_iter_against_another_model):
            begin_episode = self.n_iter_against_another_model*round
            end_episode = self.n_iter_against_another_model*(round+1)
            
            number_of_models_in_current_round = self.find_number_agents_in_round()
            print(f"Number of models still in round {number_of_models_in_current_round}")
            for _ in range(number_of_models_in_current_round):
                model_params = self.find_least_trained_agent()
                rival_model_params = self.find_worth_opponent(model_params)
                parameters = [model_params, rival_model_params, rival_model_params, rival_model_params, rival_model_params]
                agent_pool = AgentPool(5, self.mongodb_manager)
                agent_pool = agent_pool.create_agents_with_paths(parameters)

                A2CScum(5, agent_pool, self.n_iter_against_another_model).learn(begin_episode, end_episode)

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
        list_of_params = self.mongodb_manager.find_many(C.NAME_COLLECTION_MODELS_PARAMS)
        min_episode = min([params["current_episode"] for params in list_of_params])
        model_params = [params for params in list_of_params if params["current_episode"] == min_episode]
        model_param: Dict = np.random.choice(model_params)
        model_param.pop('_id')
        model_param['training'] = True
        return model_param
    
    def find_number_agents_in_round(self) -> Dict[str, Any]:
        list_of_params = self.mongodb_manager.find_many(C.NAME_COLLECTION_MODELS_PARAMS)
        min_episode = min([params["current_episode"] for params in list_of_params])
        model_params = [params for params in list_of_params if params["current_episode"] == min_episode]
        return len(model_params)
    
    def find_worth_opponent(self, training_model) -> Dict[str, Any]:
        "The objective is to find a no-so good model."
        checkpoints = self.mongodb_manager.find_many(C.NAME_COLLECTION_CHECKPOINTS)
        candidate_checkpoints = [
            model_checkpoint
            for model_checkpoint in checkpoints
            if model_checkpoint["current_episode"] <= training_model['current_episode'] &
              model_checkpoint["current_episode"] >= training_model['current_episode']//2
        ]

        chosen_params: Dict = np.random.choice(candidate_checkpoints)
        chosen_params.pop('_id')
        return chosen_params
        

if __name__ == '__main__':
    pbt = PopulationBasedTraining()
    pbt.initialize_training()
    pbt.training()
        