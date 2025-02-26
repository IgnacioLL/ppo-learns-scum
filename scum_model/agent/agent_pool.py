import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from agent.a2c_agent import A2CAgent
from typing import List
import numpy as np
from config.constants import Constants as C

from utils import utils

class AgentPool:
    def __init__(self, num_agents: int, load_path: str=None, **kwargs):
        self.number_of_agents = num_agents
        self.agents = self._create_agents(load_path, **kwargs)
        self.order = list(range(self.number_of_agents))
        self.previous_order = self.order.copy()
        self.previous_agents = self.agents.copy()
        self.best_reward = 0
        self.worst_reward = 0
        self.kwargs = kwargs


    def randomize_order(self):
        self.agents = utils.shuffle_list(self.agents)

    def get_which_agent_training(self):
        for agent_number in range(self.number_of_agents):
            if self.agents[agent_number].training:
                return agent_number


    def refresh_agents_with_previous_executions(self) -> List[A2CAgent]:
        os.makedirs(C.MODELS_PATH, exist_ok=True)
        files = os.listdir(f"{C.MODELS_PATH}")
        executed_episodes = [int(file.split("_")[1][:-3]) for file in files]
        executed_episodes.sort()
        executed_episodes = executed_episodes[-(self.number_of_agents-1):]
        
        if len(executed_episodes) >= (self.number_of_agents - 1):
            for agent_number, episode in zip(range(self.number_of_agents), executed_episodes):
                if self.agents[agent_number].training==False:
                    path = f"{C.MODELS_PATH}/model_{str(episode)}.pt"
                    self.agents[agent_number] = A2CAgent(number_players=self.number_of_agents, path=path, **self.kwargs)
        else:
            for agent_number, episode in zip(range(1, self.number_of_agents), executed_episodes):
                if self.agents[agent_number].training==False:
                    self.agents[agent_number] = A2CAgent(number_players=self.number_of_agents, path=None, **self.kwargs)


    def _create_agents(self, path=None, **kwargs) -> List[A2CAgent]:
        training_agent = A2CAgent(number_players=self.number_of_agents, path=path, **kwargs)
        training_agent.set_training(True)
        agents = [training_agent]
        for i in range(1, self.number_of_agents):
            agent = A2CAgent(number_players=self.number_of_agents, **kwargs)
            agents.append(agent)
        return agents
    
    def create_agents_with_paths(self, paths: List[str], models: List[str], **kwargs):
        self.agents = [A2CAgent(number_players=self.number_of_agents, path=path, model=model, **kwargs) 
                       for model, path in zip(models, paths)]
        return self


    def get_agent(self, agent_number: int) -> A2CAgent:
        return self.agents[agent_number]

    def set_agent(self, agent: A2CAgent, agent_number: int = 0) -> None:
        self.agents[agent_number] = agent

    def get_better_model(self):
        return self.get_agent(self.order[0])

    def swap_worst_models_for_best_ones(self, average_rewards: List[float], tol=0.1) -> None:
        self.update_order(average_rewards)
        self.save_agents()
        self.swap(tol=tol)


    def update_order(self, average_rewards: List[float]) -> None:
        self.previous_order = self.order.copy()
        order = np.argsort(average_rewards).tolist()
        order.reverse()
        self.order = order.copy()
        self.best_reward = max(average_rewards)
        self.worst_reward = min(average_rewards)

    def save_agents(self) -> None:
        self.previous_agents = self.agents.copy()

    def swap(self, tol) -> None:
        worst_agent = self.order[-1]
        best_agent = self.previous_order[0]
        
        if (self.best_reward - self.worst_reward) > tol:
            self.agents[worst_agent].model.load_state_dict(self.previous_agents[best_agent].model.state_dict())

    def apply_discounted_returns_in_agents_buffer(self, episode_rewards, discount):
        for agent_number, agent_rewards in enumerate(episode_rewards):
            self.agents[agent_number].buffer.apply_discounted_returns_in_buffer(agent_rewards, discount)

    
    def save_models(self, episode):
        for agent_number in range(self.number_of_agents):
            if self.agents[agent_number].training:
                self.agents[agent_number].save_model(f"{C.MODELS_PATH}/model_{str(episode)}.pt")


    def flush_agents_buffers(self, episode: int):
        for agent_number in range(self.number_of_agents):
            os.makedirs("./analytics/data/", exist_ok=True)
            self.agents[agent_number].buffer.flush_buffer(f"./analytics/data/agent_{agent_number}_buffer_episode_{episode}.parquet")
        