import torch 
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.optim import Adam

from constants import Constants as C
from collections import deque

import numpy as np
import pandas as pd
from typing import List

import torch.nn as nn
import random

class A2C_model(nn.Module):
    def __init__(self, number_of_players):
        super().__init__()

        self.chore_part = nn.Sequential(
            nn.Linear(C.NUMBER_OF_POSSIBLE_STATES + C.NUMBER_OF_CARDS_PER_SUIT + 1 + number_of_players, 256),
            nn.LayerNorm(256),  # Use LayerNorm instead of BatchNorm1d
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),  # Use LayerNorm instead of BatchNorm1d
            nn.ReLU()
        )

        self.value_estimate = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # Use LayerNorm instead of BatchNorm1d
            nn.ReLU(), 
            nn.Linear(256, 1)
        )
        
        self.action_probability = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # Use LayerNorm instead of BatchNorm1d
            nn.ReLU(), 
            nn.Linear(256, C.NUMBER_OF_POSSIBLE_STATES)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if x.dim() == 1:  # If input is 1D, reshape to 2D by adding batch dimension
            x = x.unsqueeze(0)
    
        x = self.chore_part(x)
        value = self.value_estimate(x)
        action_probs = self.softmax(self.action_probability(x))
        return value, action_probs

        
class A2C_Agent:
    def __init__(self, learning_rate: float=1e-4, discount: float=None, number_players=C.NUMBER_OF_AGENTS, path=None):
        
        if path is None:
            self.model = A2C_model(number_players).to(C.DEVICE)
        else:
            self.model = A2C_model(number_players).to(C.DEVICE)
            self.model.load_state_dict(torch.load(path))
        self.learning_rate = learning_rate
        self.initial_learning_rate = self.learning_rate * C.INITIAL_LR_FACTOR
        self.current_learning_rate = self.initial_learning_rate
        self.discount = discount if discount is not None else C.DISCOUNT

        self.optimizer = Adam(params=self.model.parameters(), lr=self.learning_rate)
        self.memory = []
        self.eps = np.finfo(np.float32).eps.item()

    @torch.no_grad()
    def predict(self, state):
        _, probs = self.model(state)
        return probs


    def train(self, step):
        self.warmup_learning_rate(step)

        states = torch.stack([transition[0].to(C.DEVICE) for transition in self.memory])
        rewards = [transition[1] for transition in self.memory]
        new_states = torch.stack([transition[2].to(C.DEVICE) for transition in self.memory])
        self.memory = []

        returns = []
        G=0
        for reward in rewards[::-1]:
            G = reward + self.discount * G
            returns.insert(0, G)
        returns = torch.tensor(returns, device=C.DEVICE)

        v_value_current_state = self.model(states)[0]
        policy_log = - torch.log(self.model(states)[1])

        returns = (returns - torch.mean(returns)) / (torch.std(returns) + self.eps)

        value_errors = F.smooth_l1_loss(v_value_current_state.squeeze(),returns) ## Can be replaced for actual G_{t+1}

        v_value_new_state = self.model(new_states)[0]
        advantage = returns - v_value_current_state.squeeze() + self.discount * v_value_new_state.squeeze()

        policy_error = policy_log * advantage.unsqueeze(1)

        self.optimizer.zero_grad()

        loss = torch.mean(value_errors + torch.sum(policy_error))
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def _is_action_valid(self, prediction, action_space):
        action_to_take = Categorical(prediction).sample()
        valid = action_space[action_to_take] != 0
        return C.REWARD_CHOOSE_IMPOSIBLE_ACTION if valid == False else 0
    
    def decide_move(self, state: torch.tensor) -> int:
        action_space = state[:C.NUMBER_OF_POSSIBLE_STATES]
        if state is None:
            state = torch.tensor([0 for _ in range((C.NUMBER_OF_POSSIBLE_STATES - 1) + self.number_players + C.NUMBER_OF_CARDS_PER_SUIT+1)] + [1], dtype=torch.float32).to(C.DEVICE)
        else:
            prediction = self.predict(state)
            # We set a large negative value to the masked predictions that are not possible
            masked_predictions = prediction[0] * action_space
            masked_predictions_scaled = masked_predictions / torch.sum(masked_predictions)

            if int(random.random()*1000) % 500 == 0: 
                print("State is: ", state)
                print("Prediction is: ", prediction)
                print("Masked prediction is: ", masked_predictions)

            prediction_masked = Categorical(masked_predictions_scaled)
            
            rw_decision = self._is_action_valid(prediction, action_space)

            return  (prediction_masked.sample() + 1).item(), rw_decision ## esto devolvera un valor entre 1 y 57 que sera la eleccion del modelo
        
    def save_model(self, path: str = "model.pt") -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str = "model.pt") -> nn.Module:
        model = torch.load(path)
        return model

    def warmup_learning_rate(self, step):
        if step < C.WARMUP_STEPS:
            self.current_learning_rate = self.initial_learning_rate + (self.learning_rate - self.initial_learning_rate) * (step / C.WARMUP_STEPS)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_learning_rate
        elif step == C.WARMUP_STEPS:
            self.current_learning_rate = self.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_learning_rate

    

class Pool_A2C_Agents:
    def __init__(self, num_agents: int, load_checkpoints: bool = False, **kwargs):
        self.number_of_agents = num_agents
        self.agents = self._create_agents(load_checkpoints, **kwargs)
        self.order = list(range(self.number_of_agents))
        self.previous_order = self.order.copy()
        self.previous_agents = self.agents.copy()

    def _create_agents(self, load_checkpoints: bool, **kwargs):
        agents = []
        for i in range(self.number_of_agents):
            agent_kwargs = kwargs.copy()
            if load_checkpoints:
                agent_kwargs['path'] = f"{C.CHECKPOINTS_PATH}/agent_{i+1}.pt"
            agent = A2C_Agent(n=self.number_of_agents, **agent_kwargs)
            agents.append(agent)
        return agents

    def get_agent(self, agent_number):
        return self.agents[agent_number]

    
    def update_order(self, average_rewards: List[float]) -> None:
        self.previous_order = self.order.copy()
        order = np.argsort(average_rewards).tolist()
        order.reverse()
        self.order = order.copy()
        
    def save_agents(self) -> None:
        self.previous_agents = self.agents.copy()

    def refresh_agents(self) -> None:
        # Load the weights of the best performing agents into the worst performing ones
        worst_agent, second_worst_agent = self.order[-1], self.order[-2]
        best_agent, second_best_agent = self.previous_order[0], self.previous_order[1]

        print(f"Refreshing agents {worst_agent} and {second_worst_agent} with {best_agent} and {second_best_agent}")

        self.agents[worst_agent].model.load_state_dict(self.previous_agents[best_agent].model.state_dict())
        self.agents[second_worst_agent].model.load_state_dict(self.previous_agents[second_best_agent].model.state_dict())




        

