import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


import torch 
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.optim import Adam
import gc

from config.constants import Constants as C

import numpy as np

import torch.nn as nn
import random

from nnet.nnet import NNet

from utils.ml_utils import compute_grad_stats, compute_discounted_returns, WarmupLRScheduler
from utils.utils import compact_form_of_states

from typing import Union

class A2CAgent:
    def __init__(self, training: bool=False, learning_rate: float = 1e-4, discount: float = None, number_players: int = C.NUMBER_OF_AGENTS, path: str = None, model="big", model_id=None, entropy_coef=0, policy_error_coef=1, value_error_coef=1, epochs=C.N_EPOCH_PER_STEP):
        self.model = NNet(number_of_players=number_players, model=model, id=model_id).to(C.DEVICE)
        if path:
            self.model.load_state_dict(torch.load(path, weights_only=False))

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
        self.buffer = {'states': [], 'rewards': [], 'returns': [], 'action_space': [], 'actions': [], 'old_log_probs': []}
        self.eps = np.finfo(np.float32).eps.item()

        self.policy_error_coef = policy_error_coef
        self.value_error_coef = value_error_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.training = training

    def set_training(self, set_to_train: bool) -> None:
        self.training = set_to_train

    def save_in_buffer(self, current_state, reward, action_space, action, log_prob):
        # Detach any tensors that might have gradients
        if isinstance(current_state, torch.Tensor):
            current_state = current_state.detach().cpu()
        if isinstance(action_space, torch.Tensor):
            action_space = action_space.detach().cpu()
        if isinstance(log_prob, torch.Tensor):
            log_prob = log_prob.detach().cpu()
            
        self.buffer['states'].append(compact_form_of_states(current_state))
        self.buffer['rewards'].append(reward)
        self.buffer['action_space'].append(action_space)
        self.buffer['actions'].append(action)
        self.buffer['old_log_probs'].append(log_prob)

    @torch.no_grad()
    def predict(self, state):
        _, probs = self.model(state)
        return probs.detach().clone()  # Explicitly detach and clone


    def train(self, episode, batch_size=32):
        # Initialize metrics
        metrics = self.initialize_metrics()

        data = self.load_data_from_buffer()
        self.clear_buffer()

        data = self.remove_impossible_states(data)

        for _ in range(self.epochs):
            data = self.shuffle_data(data)

            # Iterate over batches
            for batch_data in self.create_batches(data, batch_size):
                batch_metrics = self.train_on_batch(batch_data)
                self.accumulate_metrics(metrics, batch_metrics)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Average metrics across batches
        return self.average_metrics(metrics, len(data[0])*self.epochs, batch_size)


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
            'learning_rate': 0,
            "variance_in_logits": 0,
            'mean_change_ratio': 0, 
            'max_change_ratio': 0, 
            'std_change_ratio': 0, 
            'max_prob': 0
        }
    

    def load_data_from_buffer(self):
        states = torch.stack([state.to(C.DEVICE) for state in self.buffer['states']])
        returns = torch.tensor([reward for reward in self.buffer['returns']], device=C.DEVICE)
        action_masks = torch.stack([action_space.to(C.DEVICE) for action_space in self.buffer['action_space']])
        actions = torch.tensor([action for action in self.buffer['actions']], device=C.DEVICE)
        old_log_probs = torch.tensor([log_prob for log_prob in self.buffer['old_log_probs']], device=C.DEVICE)
        
        return states, returns, action_masks, actions, old_log_probs
    

    def clear_buffer(self):
        self.buffer = {'states': [], 'rewards': [], 'returns': [], 'action_space': [], 'actions': [], 'old_log_probs': []}
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def remove_impossible_states(self, data):
        states, rewards, action_masks, actions, old_log_probs = data
        impossible_states = ~torch.all(states == 0, dim=1)
        states = states[impossible_states]
        rewards = rewards[impossible_states]
        action_masks = action_masks[impossible_states]
        old_log_probs = old_log_probs[impossible_states]

        return states, rewards, action_masks, actions, old_log_probs
    
    
    def shuffle_data(self, data):
        states, returns, action_masks, actions, old_log_probs = data
        num_transitions = len(states)

        # Generate a permutation index
        permutation = torch.randperm(num_transitions)

        # Apply the permutation to shuffle the data
        shuffled_states = states[permutation]
        shuffled_rewards = returns[permutation]
        shuffled_action_masks = action_masks[permutation]
        shuffled_actions = actions[permutation]
        shuffled_log_probs = old_log_probs[permutation]

        return shuffled_states, shuffled_rewards, shuffled_action_masks, shuffled_actions, shuffled_log_probs


    def create_batches(self, data, batch_size):
        states, returns, action_space, actions, old_log_probs = data
        num_batches = (len(states) + batch_size - 1) // batch_size
        for i in range(num_batches):
            start, end = i * batch_size, min((i + 1) * batch_size, len(states))
            yield states[start:end], returns[start:end], action_space[start:end], actions[start:end], old_log_probs[start:end]


    def train_on_batch(self, batch_data):
        batch_states, batch_returns, batch_action_space, batch_action, batch_log_prob = batch_data
        value_preds, policy_logits = self.model(batch_states)
        value_preds = value_preds.squeeze()

        masked_policy_logits = self.mask_impossible_actions(policy_logits, batch_action_space)
        masked_policy_probs = F.softmax(masked_policy_logits, dim=-1)
        cat_distribution = Categorical(masked_policy_probs)
        log_prob = cat_distribution.log_prob(batch_action-1)

        advantadge = self.compute_advantadge(batch_returns, value_preds)

        if len(advantadge) == 1:
            advantadge_norm = torch.zeros_like(advantadge)
        else:
            advantadge_norm = (advantadge - advantadge.mean()) / (advantadge.std() + 1e-10)

        policy_loss, ratio = self.compute_policy_error_with_clipped_surrogate(log_prob, batch_log_prob, advantadge_norm)
        value_loss = F.mse_loss(value_preds, batch_returns).mean()
        entropy = self.compute_entropy(masked_policy_probs, log_prob)

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
            'entropy': entropy.mean().item() * self.entropy_coef,
            'total_loss': total_loss.item(),
            'value_prediction_avg': value_preds.squeeze().mean().item(),
            'returns': batch_returns.mean().item(),
            **gradient_stats,
            'advantadge': advantadge.mean().item(), 
            'advantadge_normalized': advantadge_norm.mean().item(),
            "learning_rate": self.scheduler.get_current_learning_rate(),
            'variance_in_logits': policy_logits.std().item(),
            'mean_change_ratio': ratio.mean().item(), 
            'max_change_ratio': ratio.max().item(), 
            'std_change_ratio': ratio.std().item(), 
            'max_prob': masked_policy_probs.max().item()
        }


    def mask_impossible_actions(self, action_masks, policy_logits):
        masked_policy_logits = policy_logits.clone()
        masked_policy_logits[(action_masks==0)] = float('-inf')

        return masked_policy_logits


    def compute_advantadge(self, returns, value_preds):
        return returns - value_preds


    def compute_policy_error(self, policy_log_probs, advantages):
        policy_loss = -policy_log_probs  * advantages.unsqueeze(1).detach()
        return policy_loss.mean()


    def compute_policy_error_with_clipped_surrogate(self, policy_log_probs, old_log_probs, advantages, clip_range=.2):
        ratio = torch.exp(policy_log_probs - old_log_probs)
        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        return policy_loss.mean(), ratio


    def compute_entropy(self, policy_probs, policy_log_probs):
        return -(policy_probs * policy_log_probs.unsqueeze(1)).sum(dim=1).mean()


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


    def decide_move(self, state: torch.tensor, action_space) -> Union[int, torch.Tensor]:
        if state is None:
            state = self.create_only_pass_state()
        else:
            compact_state = compact_form_of_states(state)
            prediction = self.predict(compact_state)

            masked_predictions= self.mask_impossible_actions_in_prediction(prediction, action_space)
            masked_predictions_norm = self.normalize_probabilities(masked_predictions)

            self.log_current_state_and_prediction(state, prediction)

            prediction_masked = Categorical(masked_predictions_norm)
            action = prediction_masked.sample()
            
            log_prob = prediction_masked.log_prob(action)
            del state, prediction, compact_state, masked_predictions, masked_predictions_norm, prediction_masked
            return action.item() + 1, log_prob    ## esto devolvera un valor entre 1 y 57 que sera la eleccion del modelo


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
        returns: torch.Tensor = compute_discounted_returns(agent_rewards, self.discount)
        self.buffer['returns'] = self.buffer['returns'] + returns.cpu().numpy().tolist()