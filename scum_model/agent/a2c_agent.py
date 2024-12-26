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

from nnet.nnet import NNet

from utils.ml_utils import compute_grad_stats, compute_discounted_returns, WarmupLRScheduler
from utils.utils import compact_form_of_states

class A2CAgent:
    def __init__(self, learning_rate: float = 1e-4, discount: float = None, number_players: int = C.NUMBER_OF_AGENTS, path: str = None, model="small", model_id=None, entropy_coef=0, policy_error_coef=1, value_error_coef=1):
        self.model = NNet(number_of_players=number_players, model=model, id=model_id).to(C.DEVICE)
        if path:
            self.model.load_state_dict(torch.load(path))

        self.learning_rate = learning_rate
        self.initial_learning_rate = self.learning_rate * C.INITIAL_LR_FACTOR
        self.current_learning_rate = self.initial_learning_rate
        self.discount = discount if discount else C.DISCOUNT

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = WarmupLRScheduler(
            self.optimizer,
            warmup_steps=C.WARMUP_STEPS,
            initial_lr=learning_rate*C.INITIAL_LR_FACTOR,
            target_lr=learning_rate
        )
        self.buffer = {'states': [], 'rewards': [], 'new_states': [], 'returns': []}
        self.eps = np.finfo(np.float32).eps.item()

        self.policy_error_coef = policy_error_coef
        self.value_error_coef = value_error_coef
        self.entropy_coef = entropy_coef

    def save_in_buffer(self, current_state, reward, new_state):
        self.buffer['states'].append(current_state)
        self.buffer['rewards'].append(reward)
        self.buffer['new_states'].append(new_state)

    @torch.no_grad()
    def predict(self, state):
        _, probs = self.model(state)
        return probs
    
    def train(self, episode, batch_size=32):

        data = self.load_data_from_buffer()
        self.clear_buffer()

        data = self.remove_impossible_states(data)
        states, returns, _, action_space = self.shuffle_data(data)
        compact_states = torch.stack([compact_form_of_states(state) for state in states])

        # Initialize metrics
        metrics = self.initialize_metrics()

        # Iterate over batches
        for batch_states, batch_returns, batch_action_space in self.create_batches(compact_states, returns, action_space, batch_size):
            batch_metrics = self.train_on_batch(batch_states, batch_returns, batch_action_space)
            self.accumulate_metrics(metrics, batch_metrics)

        # Average metrics across batches
        return self.average_metrics(metrics, len(compact_states), batch_size)
    
    def load_data_from_buffer(self):
        states = torch.stack([state.to(C.DEVICE) for state in self.buffer['states']])
        returns = torch.tensor([reward for reward in self.buffer['returns']], device=C.DEVICE)
        new_states = torch.stack([new_state.to(C.DEVICE) for new_state in self.buffer['new_states']])
        action_masks = torch.stack([state[:C.NUMBER_OF_POSSIBLE_STATES].to(C.DEVICE) for state in self.buffer['states']])  # Assuming masks are stored
        
        return states, returns, new_states, action_masks
    
    def clear_buffer(self):
        self.buffer = {'states': [], 'rewards': [], 'new_states': [], 'returns': []}
    
        
    def remove_impossible_states(self, data):
        states, rewards, new_states, action_masks = data
        impossible_states = ~torch.all(states == 0, dim=1)
        states = states[impossible_states]
        rewards = rewards[impossible_states]
        new_states = new_states[impossible_states]
        action_masks = action_masks[impossible_states]

        return states, rewards, new_states, action_masks
    
    
    def shuffle_data(self, data):
        states, returns, new_states, action_masks = data
        num_transitions = len(states)

        # Generate a permutation index
        permutation = torch.randperm(num_transitions)

        # Apply the permutation to shuffle the data
        shuffled_states = states[permutation]
        shuffled_rewards = returns[permutation]
        shuffled_new_states = new_states[permutation]
        shuffled_action_masks = action_masks[permutation]

        return shuffled_states, shuffled_rewards, shuffled_new_states, shuffled_action_masks
    
    def initialize_metrics(self):
        return {
            'value_loss': 0,
            'policy_loss': 0,
            'entropy': 0,
            'total_loss': 0,
            'value_prediction_avg': 0,
            'returns': 0,
            'mean_gradient': 0,
            'median_gradient': 0,
            'max_gradient': 0,
            'min_gradient': 0,
            'p99_gradient': 0,
            'p01_gradient': 0,
            'advantadge': 0,
            'advantadge_normalized': 0,
            'learning_rate': 0
        }


    def create_batches(self, compact_states, returns, action_space, batch_size):
        num_batches = (len(compact_states) + batch_size - 1) // batch_size
        for i in range(num_batches):
            start, end = i * batch_size, min((i + 1) * batch_size, len(compact_states))
            yield compact_states[start:end], returns[start:end], action_space[start:end]


    def train_on_batch(self, batch_states, batch_returns, batch_action_space):
        value_preds, policy_logits = self.model(batch_states)
        value_preds = value_preds.squeeze()

        masked_policy_logits = self.mask_impossible_actions(policy_logits, batch_action_space)
        masked_policy_probs = F.softmax(masked_policy_logits, dim=-1)
        masked_policy_log_probs = torch.log(masked_policy_probs + 1e-10)

        advantadge = self.compute_advantadge(batch_returns, value_preds)

        policy_loss = self.compute_policy_error_with_masked_actions(masked_policy_log_probs, advantadge, batch_action_space)
        value_loss = F.mse_loss(value_preds, batch_returns).mean()
        entropy = self.compute_entropy(masked_policy_probs, masked_policy_log_probs)
        total_loss = (
            self.policy_error_coef * policy_loss +
            self.value_error_coef * value_loss -
            self.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        gradient_stats = self.get_gradient_stats()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {
            'value_loss': value_loss.item() * self.value_error_coef,
            'policy_loss': policy_loss.item() * self.policy_error_coef,
            'entropy': entropy.item() * self.entropy_coef,
            'total_loss': total_loss.item(),
            'value_prediction_avg': value_preds.squeeze().mean().item(),
            'returns': batch_returns.mean().item(),
            **gradient_stats,
            'advantadge': advantadge.mean().item(), 
            'advantadge_normalized': 0,
            "learning_rate": self.scheduler.get_current_learning_rate()
        }
    
    def mask_impossible_actions(self, action_masks, policy_logits):
        masked_policy_logits = policy_logits.clone()
        masked_policy_logits[(action_masks==0)] = float('-inf')

        return masked_policy_logits
    
    def compute_advantadge(self, returns, value_preds):
        return returns - value_preds
    
    def compute_policy_error_with_masked_actions(self, policy_log_probs, advantages, action_space):
        policy_loss = -(policy_log_probs * action_space) * advantages.unsqueeze(1).detach()
        return policy_loss.mean()
    
    def compute_entropy(self, policy_probs, policy_log_probs):
        return -(policy_probs * policy_log_probs).sum(dim=1).mean()
    
    def accumulate_metrics(self, total_metrics, batch_metrics):
        for key in total_metrics:
            total_metrics[key] += batch_metrics[key]

    def average_metrics(self, metrics, total_samples, batch_size):
        num_batches = (total_samples + batch_size - 1) // batch_size
        return {key: value / num_batches for key, value in metrics.items()}

    def get_gradient_stats(self):
        median_grad, mean_grad, max_grad, min_grad, p99_grad, p01_grad = compute_grad_stats(self.model)
        return {
            'mean_gradient': mean_grad.item(),
            'median_gradient': median_grad.item(),
            'max_gradient': max_grad.item(),
            'min_gradient': min_grad.item(),
            'p99_gradient': p99_grad.item(),
            'p01_gradient': p01_grad.item(),
        }
    
    def _is_action_valid(self, prediction, action_space):
        action_to_take = Categorical(prediction).sample()
        valid = action_space[action_to_take] != 0
        return C.REWARD_CHOOSE_IMPOSIBLE_ACTION if valid == False else 0
    
    def decide_move(self, state: torch.tensor) -> int:
        action_space = state[:C.NUMBER_OF_POSSIBLE_STATES]
        if state is None:
            state = self.create_only_pass_state()
        else:
            compact_state = compact_form_of_states(state)
            prediction = self.predict(compact_state)

            masked_predictions= self.mask_impossible_actions_in_prediction(prediction, action_space)
            masked_predictions_norm = self.normalize_probabilities(masked_predictions)

            self.log_current_state_and_prediction(state, prediction)

            prediction_masked = Categorical(masked_predictions_norm)
            
            return  (prediction_masked.sample() + 1).item() ## esto devolvera un valor entre 1 y 57 que sera la eleccion del modelo
        
    def save_model(self, path: str = "model.pt") -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str = "model.pt") -> nn.Module:
        model = torch.load(path)
        return model

    def create_only_pass_state(self):
        return torch.tensor([0 for _ in range((C.NUMBER_OF_POSSIBLE_STATES - 1) + self.number_players + C.NUMBER_OF_CARDS_PER_SUIT+1)] + [1],
                             dtype=torch.float32).to(C.DEVICE)
    
    def log_current_state_and_prediction(self, state, prediction):
        if int(random.random()*100_000) % 50_000 == 0:
            print("State is: ", state)
            print("Prediction: ", prediction[0])

    def mask_impossible_actions_in_prediction(self, prediction, action_space):
        masked_predictions = prediction[0] * action_space
        return masked_predictions
    
    def normalize_probabilities(self, probabilities: torch.tensor):
        masked_predictions_norm = probabilities / torch.sum(probabilities)
        return masked_predictions_norm
    
    def apply_discounted_returns_in_buffer(self, agent_rewards):
        returns: torch.tensor = compute_discounted_returns(agent_rewards, self.discount)
        self.buffer['returns'] = self.buffer['returns'] + returns.cpu().numpy().tolist()

        

    


