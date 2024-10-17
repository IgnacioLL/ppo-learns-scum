import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from agent.a2c_agent import A2CAgent
from typing import List
import numpy as np
from config.constants import Constants as C

class AgentPool:
    def __init__(self, num_agents: int, load_checkpoints: bool = False, load_eval: bool = False, **kwargs):

        self.number_of_agents = num_agents
        self.agents = self._create_agents(load_checkpoints, load_eval, **kwargs)
        self.order = list(range(self.number_of_agents))
        self.previous_order = self.order.copy()
        self.previous_agents = self.agents.copy()



    def _create_agents(self, load_checkpoints: bool, load_eval: bool, **kwargs):
        agents = []
        for i in range(self.number_of_agents):
            agent_kwargs = kwargs.copy()
            if load_checkpoints:
                agent_kwargs['path'] = f"{C.CHECKPOINTS_PATH}/agent_{i+1}.pt"
            if load_eval:
                agent_kwargs['path'] = f"best_models/agent_1.pt"
            agent = A2CAgent(number_players=self.number_of_agents, **agent_kwargs)
            agents.append(agent)
        return agents

    def get_agent(self, agent_number):
        return self.agents[agent_number]

    def set_agent(self, agent: A2CAgent, agent_number: int = 0) -> None:
        self.agents[agent_number] = agent

    def get_better_model(self):
        return self.get_agent(self.order[0])

    def update_order(self, average_rewards: List[float]) -> None:
        self.previous_order = self.order.copy()
        order = np.argsort(average_rewards).tolist()
        order.reverse()
        self.order = order.copy()

    def save_agents(self) -> None:
        self.previous_agents = self.agents.copy()

    def refresh_agents(self) -> None:
        worst_agent, second_worst_agent = self.order[-1], self.order[-2]
        best_agent, second_best_agent = self.previous_order[0], self.previous_order[1]
        print(f"Refreshing agents {worst_agent} and {second_worst_agent} with {best_agent} and {second_best_agent}")
        self.agents[worst_agent].model.load_state_dict(self.previous_agents[best_agent].model.state_dict())
        self.agents[second_worst_agent].model.load_state_dict(self.previous_agents[second_best_agent].model.state_dict())