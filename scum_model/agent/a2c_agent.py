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

from utils.ml_utils import compute_grad_stats

class A2CAgent:
    def __init__(self, learning_rate: float = 1e-4, discount: float = None, number_players: int = C.NUMBER_OF_AGENTS, path: str = None, model="small", model_id=None, entropy_coef=0, policy_error_coef=.1):
        self.model = A2CModel(number_of_players=number_players, model=model, id=model_id).to(C.DEVICE)
        if path:
            self.model.load_state_dict(torch.load(path))

        self.learning_rate = learning_rate
        self.initial_learning_rate = self.learning_rate * C.INITIAL_LR_FACTOR
        self.current_learning_rate = self.initial_learning_rate
        self.discount = discount if discount else C.DISCOUNT

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.buffer = []
        self.eps = np.finfo(np.float32).eps.item()

        self.policy_error_coef = policy_error_coef
        self.entropy_coef = entropy_coef

    def save_in_buffer(self, current_state, reward, new_state):
        self.buffer.append((current_state, reward, new_state))

    def clear_buffer(self):
        self.buffer = []


    @torch.no_grad()
    def predict(self, state):
        _, probs = self.model(state)
        return probs
    
    def train(self, step):

        data = self.load_data_from_buffer()
        states, rewards, _, action_masks = self.remove_impossible_states(data)

        self.clear_buffer()
        returns = self.compute_discounted_returns(rewards)
        
        value_preds, policy_logits = self.model(states)

        masked_policy_logits = self.mask_impossible_actions(policy_logits, action_masks)
        
        masked_policy_probs = F.softmax(masked_policy_logits, dim=-1)

        eps = 1e-10
        masked_policy_log_probs = torch.log(masked_policy_probs + eps)

        advantadge = self.compute_advantadge(returns, value_preds)
        policy_loss = self.compute_policy_error_with_masked_actions(masked_policy_log_probs,
                                                                          advantadge, action_masks)
        value_errors = F.smooth_l1_loss(value_preds.squeeze(), returns)

        self.optimizer.zero_grad()
        
        # Scale losses appropriately
        value_loss = value_errors.mean()
        
        # Add entropy bonus to encourage exploration
        entropy = self.compute_entropy(masked_policy_probs, masked_policy_log_probs)
        
        # Combine losses with appropriate scaling for dealing with inestability in training
        total_loss = self.policy_error_coef * policy_loss +  value_loss - self.entropy_coef * entropy
        
        # Gradient clipping
        total_loss.backward()
        median_grad, mean_grad, max_grad, min_grad, p99_grad, p01_grad = compute_grad_stats(self.model)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()

        return {
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item() * self.policy_error_coef,
            'entropy': entropy.item() * self.entropy_coef,
            'total_loss': total_loss.item(), 
            'value_prediction_avg': value_preds.squeeze().mean().item(),
            'returns': returns.mean().item(),
            'mean_gradient': mean_grad.item(),
            'median_gradient': median_grad.item(),
            'max_gradient': max_grad.item(),
            'min_gradient': min_grad.item(),
            'p99_gradient': p99_grad.item(),
            'p01_gradient': p01_grad.item(),
            'advantadge': advantadge.mean().item()
        }
    
    def load_data_from_buffer(self):
        states = torch.stack([transition[0].to(C.DEVICE) for transition in self.buffer])
        rewards = torch.tensor([transition[1] for transition in self.buffer], device=C.DEVICE)
        new_states = torch.stack([transition[2].to(C.DEVICE) for transition in self.buffer])
        action_masks = torch.stack([transition[0][:C.NUMBER_OF_POSSIBLE_STATES].to(C.DEVICE) for transition in self.buffer])  # Assuming masks are stored
        
        return states, rewards, new_states, action_masks
    
    def remove_impossible_states(self, data):
        states, rewards, new_states, action_masks = data
        impossible_states = ~torch.all(states == 0, dim=1)
        states = states[impossible_states]
        rewards = rewards[impossible_states].cpu().tolist()
        new_states = new_states[impossible_states]
        action_masks = action_masks[impossible_states]

        return states, rewards, new_states, action_masks
    
    def compute_discounted_returns(self, rewards: list) -> torch.tensor:
        G = 0
        returns = []
        for reward in rewards[::-1]:
            G = reward + self.discount * G
            returns.insert(0, G)
        return torch.tensor(returns, device=C.DEVICE)
        
    def mask_impossible_actions(self, action_masks, policy_logits):
        masked_policy_logits = policy_logits.clone()
        masked_policy_logits[(action_masks==0)] = float('-inf')

        return masked_policy_logits
    
    def compute_advantadge(self, returns, value_preds):
        return returns - value_preds.squeeze()
    
    def compute_policy_error_with_masked_actions(self, policy_log_probs, advantages, action_masks):
        policy_loss = (-policy_log_probs * action_masks) * advantages.unsqueeze(1)

        return policy_loss.mean()
    
    def compute_entropy(self, policy_probs, policy_log_probs):
        return -(policy_probs * policy_log_probs).sum(dim=1).mean()
        
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

            if int(random.random()*10_000) % 5_000 == 0:
                print("State is: ", state)
                print("Prediction: ", prediction[0])

            prediction_masked = Categorical(masked_predictions_scaled)
            
            return  (prediction_masked.sample() + 1).item() ## esto devolvera un valor entre 1 y 57 que sera la eleccion del modelo
        
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
