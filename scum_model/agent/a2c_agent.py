import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


import torch 
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.optim import Adam

from config.constants import Constants as C

import numpy as np

import torch.nn as nn
import random

from agent.a2c_model import A2CModel

class A2CAgent:
    def __init__(self, learning_rate: float = 1e-4, discount: float = None, number_players: int = C.NUMBER_OF_AGENTS, path: str = None, model="small", model_id=None, entropy_coef=0, policy_error_coef=.1):
        self.model = A2CModel(number_of_players=number_players, model=model, id=model_id).to(C.DEVICE)
        if path:
            print(f"Loading model from {path}")
            self.model.load_state_dict(torch.load(path))

        self.learning_rate = learning_rate
        self.initial_learning_rate = self.learning_rate * C.INITIAL_LR_FACTOR
        self.current_learning_rate = self.initial_learning_rate
        self.discount = discount if discount else C.DISCOUNT

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.memory = []
        self.eps = np.finfo(np.float32).eps.item()

        self.policy_error_coef = policy_error_coef
        self.entropy_coef = entropy_coef

    @torch.no_grad()
    def predict(self, state):
        _, probs = self.model(state)
        return probs

    def train(self, step):
        # Stack states and prepare data as before
        states = torch.stack([transition[0].to(C.DEVICE) for transition in self.memory])
        rewards = torch.tensor([transition[1] for transition in self.memory], device=C.DEVICE)
        new_states = torch.stack([transition[2].to(C.DEVICE) for transition in self.memory])
        # Get action masks if they're part of your transitions
        action_masks = torch.stack([transition[0][:C.NUMBER_OF_POSSIBLE_STATES].to(C.DEVICE) for transition in self.memory])  # Assuming masks are stored
        
        filter_avoid_impossible_state = ~torch.all(states == 0, dim=1)
        states = states[filter_avoid_impossible_state]
        rewards = rewards[filter_avoid_impossible_state].cpu().tolist()
        new_states = new_states[filter_avoid_impossible_state]
        action_masks = action_masks[filter_avoid_impossible_state]

        self.memory = []
        
        # Calculate returns as before
        returns = []
        G = 0
        for reward in rewards[::-1]:
            G = reward + self.discount * G
            returns.insert(0, G)
        returns = torch.tensor(returns, device=C.DEVICE)
        
        # Get model predictions
        value_preds, policy_logits = self.model(states)
        
        # Apply temperature scaling to logits (optional, can help with stability)
        temperature = 1.0
        policy_logits = policy_logits / temperature
        
        # Apply mask by setting logits of invalid actions to large negative values
        masked_policy_logits = policy_logits.clone()
        masked_policy_logits[(action_masks==0)] = float('-inf')
        
        # Get probabilities using softmax only on valid actions
        policy_probs = F.softmax(masked_policy_logits, dim=-1)
        
        # Calculate policy log probabilities, avoiding log(0)
        # Add small epsilon to prevent taking log of zero
        eps = 1e-10
        policy_log_probs = torch.log(policy_probs + eps)
        
        # Value loss
        print("Value prediction average: ", value_preds.squeeze().mean())
        print("Returns prediction average: ", returns.mean())

        value_errors = F.smooth_l1_loss(value_preds.squeeze(), returns)
        
        # Calculate advantages
        advantages = returns - value_preds.squeeze()
        
        # Policy loss - only consider valid actions
        # Multiply by mask to zero out invalid actions

        policy_error = (policy_log_probs * action_masks) * advantages.unsqueeze(1)
        
        # Calculate mean policy error only over valid actions
        valid_actions_count = action_masks.sum(dim=1, keepdim=True)
        policy_error = policy_error.sum(dim=1) / valid_actions_count.squeeze()
        
        # Optimize
        self.optimizer.zero_grad()
        
        # Scale losses appropriately
        value_loss = value_errors.mean()
        policy_loss = -policy_error.mean()  # Negative because we want to maximize
        
        # Add entropy bonus to encourage exploration (optional)
        entropy = -(policy_probs * policy_log_probs).sum(dim=1).mean()
        entropy_coef = 0  # Adjust this coefficient as needed
        
        # Combine losses with appropriate scaling
        value_coef = 1  # Adjust this coefficient as needed
        policy_coef = 0.1  # Adjust this coefficient as needed
        total_loss = self.policy_error_coef * policy_loss + value_coef * value_loss - self.entropy_coef * entropy
        
        # Gradient clipping
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()

        return {
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }
    
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
