import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from agent.agent import Agent
from typing import List, Dict, Any
from config.constants import Constants as C
from db.db import MongoDBManager
import torch
from utils import utils

from typing import Union

class AgentPool:
    def __init__(self, num_agents: int, mongodb_manager: MongoDBManager=None):
        self.number_of_agents = num_agents
        self.order = list(range(self.number_of_agents))
        self.mongodb_manager = mongodb_manager

    def __del__(self):
        """Destructor that ensures proper cleanup of all agents in the pool when garbage collected."""
        try:
            if hasattr(self, 'agents'):
                for agent in self.agents:
                    if agent is not None:
                        # This will trigger each agent's __del__ method
                        del agent
                
                # Clear the agents list
                self.agents = []
        except Exception as e:
            print(f"Error during AgentPool cleanup: {e}")

    def get_agent(self, agent_number: int) -> Agent:
        return self.agents[agent_number]

    def set_agent(self, agent: Agent, agent_number: int = 0) -> None:
        self.agents[agent_number] = agent

    def randomize_order(self):
        self.agents = utils.shuffle_list(self.agents)
        return self

    def get_which_agent_training(self):
        for agent_number in range(self.number_of_agents):
            if self.agents[agent_number].training:
                return agent_number

    def create_agents_with_nnet_initialization(self, **model_params) -> List[Agent]:
        training_agent = Agent(**model_params, training=True)
        self.agents = [training_agent]
        for i in range(1, self.number_of_agents):
            agent = Agent(**model_params)
            self.agents.append(agent)
        return self
    
    def create_agents_with_parameters(self, list_of_params_agent: Union[List[Dict[str, Any]], Dict[str, Any]]):
        if isinstance(list_of_params_agent, list):
            self.agents = [Agent(**params) for params in list_of_params_agent]
        elif isinstance(list_of_params_agent, dict):
            self.agents = [Agent(**list_of_params_agent) for _ in range(self.number_of_agents)]
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

    def extract_wins_agents_with_invented_id(self):
        agents_wins = {}
        for agent_number in range(self.number_of_agents):
            agent = self.get_agent(agent_number)
            agents_wins[agent.model_id + "-" + str(agent_number)] = sum(agent.wins)
        return agents_wins
    
    def extract_wins_agents_w_current_episode(self):
        agents_wins = {}
        for agent_number in range(self.number_of_agents):
            agent = self.get_agent(agent_number)
            agents_wins[agent.model_id + "-" + str(agent.current_episode)] = sum(agent.wins)
        return agents_wins
    
    def add_next_actions_in_agents_buffers(self, next_actions: Dict[int, List[int]]):
        for agent_number in next_actions.keys():
            self.agents[agent_number].buffer.add_next_actions(next_actions[agent_number])

    def clear_buffer_all_agents(self):
        for agent in self.agents:
            agent.buffer.clear_buffer()

    def cleanup(self):
        """Explicitly delete models and clear agent list to release resources."""
        for agent in self.agents:
            if hasattr(agent, 'cleanup'):
                agent.cleanup()  # Use the agent's cleanup method
            else:
                # Fallback for backward compatibility
                if hasattr(agent, 'model') and agent.model is not None:
                    del agent.model # Remove reference to the model
                if hasattr(agent, 'actor') and agent.actor is not None:
                    del agent.actor
                if hasattr(agent, 'critic') and agent.critic is not None:
                    del agent.critic
                # Add deletion for any other large objects (like optimizers if they exist)
                if hasattr(agent, 'optimizer') and agent.optimizer is not None:
                    del agent.optimizer # Optimizers can hold references to model params

        # Clear the list of agents itself
        self.agents = []
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Force CUDA cache clear
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("AgentPool cleanup complete.") # Debug print