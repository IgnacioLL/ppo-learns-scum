import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from agent.a2c_agent import A2CAgent
from typing import List, Dict, Any
from config.constants import Constants as C
from db.db import MongoDBManager

from utils import utils

from typing import Union

class AgentPool:
    def __init__(self, num_agents: int, mongodb_manager: MongoDBManager=None):
        self.number_of_agents = num_agents
        self.order = list(range(self.number_of_agents))
        self.mongodb_manager = mongodb_manager

    def get_agent(self, agent_number: int) -> A2CAgent:
        return self.agents[agent_number]

    def set_agent(self, agent: A2CAgent, agent_number: int = 0) -> None:
        self.agents[agent_number] = agent

    def randomize_order(self):
        self.agents = utils.shuffle_list(self.agents)
        return self

    def get_which_agent_training(self):
        for agent_number in range(self.number_of_agents):
            if self.agents[agent_number].training:
                return agent_number

    def create_agents_with_nnet_initialization(self, **model_params) -> List[A2CAgent]:
        training_agent = A2CAgent(**model_params, training=True)
        self.agents = [training_agent]
        for i in range(1, self.number_of_agents):
            agent = A2CAgent(**model_params)
            self.agents.append(agent)
        return self
    
    def create_agents_with_paths(self, list_of_params_agent: Union[List[Dict[str, Any]], Dict[str, Any]]):
        if isinstance(list_of_params_agent, list):
            self.agents = [A2CAgent(**params) for params in list_of_params_agent]
        elif isinstance(list_of_params_agent, dict):
            self.agents = [A2CAgent(**list_of_params_agent) for _ in range(self.number_of_agents)]
        return self

    def apply_discounted_returns_in_agents_buffer(self, episode_rewards, discount):
        for agent_number, agent_rewards in enumerate(episode_rewards):
            self.agents[agent_number].buffer.apply_discounted_returns_in_buffer(agent_rewards, discount)
    
    def save_models(self, episode):
        for agent_number in range(self.number_of_agents):
            agent = self.get_agent(agent_number)
            if agent.training:
                path = f"{C.MODELS_PATH}model_{agent.model_id}_{str(episode)}.pt"
                self.agents[agent_number].save_model(path)

                data = {'model_id': agent.model_id, 'load_model_path': path, 'current_episode': episode, 'model_tag': agent.model_tag, 'model_size': agent.model_size}
                if self.mongodb_manager:
                    self.mongodb_manager.insert_one(C.NAME_COLLECTION_CHECKPOINTS, data)

    def append_rewards_to_historic_record_to_each_agent(self, episode_rewards: List[float]):
        for agent_number, reward in enumerate(episode_rewards):
            self.get_agent(agent_number).append_episode_rewards(reward)
        
    def append_win_to_historic_record_to_each_agent(self, wins: List[float]):
        for agent_number, win in enumerate(wins):
            self.get_agent(agent_number).append_win(win)
        return self

    def flush_average_reward_to_tensorboard_from_each_agent(self, n_episodes, episode):
        for agent_number in range(self.number_of_agents):
            if self.get_agent(agent_number).training:
                self.get_agent(agent_number).flush_average_reward_to_tensorboard(n_episodes, episode)

    def flush_win_rate_to_tensorboard_from_each_agent(self, n_episodes, episode):
        for agent_number in range(self.number_of_agents):
            if self.get_agent(agent_number).training:
                self.get_agent(agent_number).flush_average_win_rate_to_tensorboard(n_episodes, episode)

    def extract_wins_agents(self):
        agents_wins = {}
        for agent_number in range(self.number_of_agents):
            agent = self.get_agent(agent_number)
            agents_wins[agent.model_id] = sum(agent.wins)
        return agents_wins