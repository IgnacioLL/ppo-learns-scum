import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


import torch 
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from config.constants import Constants as C

import numpy as np
import pandas as pd

import torch.nn as nn

from nnet.nnet import NNet

from utils import data_utils, logging, loss_utils, utils, env_utils
from utils.ml_utils import CoolerLRScheduler
from buffer.buffer import Buffer

from typing import Union

class A2CAgent:
    def __init__(
            self,
            model_id, 
            model_size,
            model_tag,
            training: bool=False,
            playing: bool=False,
            learning_rate: float = 1e-5, 
            discount: float = C.DISCOUNT, 
            number_players: int = C.NUMBER_OF_AGENTS, 
            load_model_path: str = None, 
            value_error_coef=1, 
            policy_error_coef=1, 
            entropy_coef=0, 
            epochs=C.N_EPOCH_PER_STEP,
            current_episode=0
            ):
        
        self.model_id = model_id
        self.model_tag = model_tag
        self.model_size = model_size
        self.current_episode = current_episode
        self.model = NNet(number_of_players=number_players, model=model_size, model_id=model_id).to(C.DEVICE)
        if load_model_path:
            print(f"Loading model from {load_model_path}")
            self.model.load_state_dict(torch.load(load_model_path, weights_only=False))

        self.learning_rate = learning_rate
        self.initial_learning_rate = self.learning_rate * C.INITIAL_LR_FACTOR
        self.current_learning_rate = self.initial_learning_rate
        self.discount = discount
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = CoolerLRScheduler(
            self.optimizer,
            cooler_steps=C.COOLER_STEPS,
            initial_lr=learning_rate*C.INITIAL_LR_FACTOR,
            target_lr=learning_rate
        )
        self.policy_error_coef = policy_error_coef
        self.value_error_coef = value_error_coef
        self.entropy_coef = entropy_coef

        self.epochs = epochs
        
        self.training = training
        self.playing = playing
        self.buffer = Buffer()
        if training:
            self.writer = SummaryWriter(log_dir=f"./runs/{model_id}-{model_tag}")

        self.episode_rewards = []
        self.wins = []

    def set_training(self, set_to_train: bool) -> None:
        self.training = set_to_train

    def set_playing(self, set_to_play: bool) -> None:
        self.playing = set_to_play

    def decide_move(self, state: torch.Tensor, action_space: torch.Tensor) -> Union[int, torch.Tensor]:
        if self.playing:
            return self.decide_move_human(state, action_space)
        else:
            return self.decide_move_model(state, action_space)
        
    def decide_move_human(self, state: torch.Tensor, action_space: torch.Tensor) -> Union[int, torch.Tensor]:
        action_space = action_space.cpu().detach().numpy()
        possible_actions = np.where(action_space == 1)[0] + 1
        print("The possible actions are:\n_______\n")
        for action in possible_actions:
            print(f"Action: {action}")
            env_utils.decode_action(action-1)

        print("Which action to take: ")
        answer = int(input())
        return answer, 1


    def decide_move_model(self, state: torch.Tensor, action_space: torch.Tensor) -> Union[int, torch.Tensor]:
        if state is None:
            state = data_utils.create_only_pass_state()
        else:
            compact_state = data_utils.compact_form_of_states(state)
            prediction = self.predict(compact_state)

            action_space = action_space.unsqueeze(0)
            masked_predictions = data_utils.mask_impossible_actions(action_space, prediction)
            masked_predictions_prob = F.softmax(masked_predictions, dim=-1)

            prediction_masked = Categorical(masked_predictions_prob)
            action = prediction_masked.sample()
            
            log_prob = prediction_masked.log_prob(action)
            return action.item() + 1, log_prob    ## esto devolvera un valor entre 1 y 57 que sera la eleccion del modelo

    @torch.no_grad()
    def predict(self, state):
        _, probs = self.model.forward(state)
        return probs.detach().clone()  # Explicitly detach and clone


    def train(self, episode, batch_size=32):
        metrics = logging.initialize_metrics()
        data = self.buffer.load_data_from_buffer()
        self.buffer.clear_buffer()

        data = data_utils.remove_impossible_states(data)
        for epoch in range(self.epochs):
            self.epoch = epoch
            data = data_utils.shuffle_data(data)

            for batch_data in data_utils.create_batches(data, batch_size):
                batch_metrics = self.train_on_batch(batch_data)
                logging.accumulate_metrics(metrics, batch_metrics)
        
        avg_metrics = logging.average_metrics(metrics)
        logging.flush_performance_stats_tensorboard(self.writer, avg_metrics, episode)

    def train_on_batch(self, batch_data):
        batch_states, batch_returns, batch_action_space, batch_action, batch_old_log_prob = batch_data
        value_preds, policy_logits = self.model.forward(batch_states)
        probs = F.softmax(policy_logits, dim=-1)
        value_preds = value_preds.squeeze()
        
        pd.DataFrame(policy_logits.clone().detach().cpu().numpy()).to_parquet("./analytics/data/policy_logits.parquet")

        masked_policy_logits = data_utils.mask_impossible_actions(batch_action_space, policy_logits)
        masked_policy_probs = F.softmax(masked_policy_logits, dim=-1)
        cat_distribution = Categorical(masked_policy_probs)
        log_prob = cat_distribution.log_prob(batch_action-1)

        advantadge = data_utils.compute_advantadge(batch_returns, value_preds)

        if len(advantadge) == 1:
            advantadge_norm = torch.zeros_like(advantadge)
        else:
            advantadge_norm = (advantadge - advantadge.mean()) / (advantadge.std() + 1e-10)

        policy_loss, ratio = loss_utils.compute_policy_error_with_clipped_surrogate(log_prob, batch_old_log_prob, advantadge_norm)
        value_loss = F.l1_loss(value_preds, batch_returns, reduction='none')
        entropy = loss_utils.compute_entropy(masked_policy_probs, log_prob)

        policy_loss = policy_loss.mean()
        value_loss = value_loss.mean()

        total_loss = (
            self.policy_error_coef * policy_loss +
            self.value_error_coef * value_loss +
            self.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        gradient_stats = logging.get_gradient_stats(self.model)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        ratio_5th_epoch = torch.abs(ratio - 1).mean().item() if self.epoch == 4 else None

        return {
            'loss_value': value_loss.item() * self.value_error_coef,
            'loss_policy': policy_loss.item() * self.policy_error_coef,
            'loss_entropy': entropy.mean().item() * self.entropy_coef,
            'loss_total': total_loss.item(),
            'value_prediction_avg': value_preds.squeeze().mean().item(),
            'returns': batch_returns.mean().item(),
            'ratio': (ratio - 1).mean().item(),
            'ratio_abs': torch.abs(ratio - 1).mean().item(),
            'ratio_5th_epoch': ratio_5th_epoch,
            'ratio_max_change': (ratio - 1).max().item(), 
            'ratio_min_change': (ratio - 1).min().item(), 
            **gradient_stats,
            'advantadge': advantadge.mean().item(), 
            'advantadge_normalized': advantadge_norm.mean().item(),
            "learning_rate": self.scheduler.get_current_learning_rate(),
            'prob_max': probs.max(dim=1)[0].mean().item(),
            'prob_2nd': utils.take_n_highest_tensor(probs, 2),
            'prob_3rd': utils.take_n_highest_tensor(probs, 3),
            'prob_median': probs.median(dim=1)[0].mean().item(),
            'prob_min': probs.min(dim=1)[0].mean().item(),
        }



    def save_model(self, path: str = "model.pt") -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str = "model.pt") -> nn.Module:
        model = torch.load(path)
        return model

    def append_episode_rewards(self, reward):
        self.episode_rewards.append(reward)
    
    def append_win(self, win):
        self.wins.append(win)

    def get_average_reward_last_n_episodes(self, n_episodes=C.AGGREGATE_STATS_EVERY) -> float:
        recent_rewards = self.episode_rewards[-n_episodes:]
        return sum(recent_rewards) / len(recent_rewards)

    def get_win_rate_last_n_episodes(self, n_episodes=C.AGGREGATE_STATS_EVERY) -> float:
        wins = self.wins[-n_episodes:].copy()
        return sum(wins) / len(wins)
    
    def flush_average_win_rate_to_tensorboard(self, n_episodes, episode):
        win_rate = self.get_win_rate_last_n_episodes(n_episodes)
        logging.flush_average_win_rate_to_tensorboard(self.writer, win_rate, episode)

    def flush_average_reward_to_tensorboard(self, n_episodes, episode):
        average_reward = self.get_average_reward_last_n_episodes(n_episodes)
        logging.flush_average_reward_to_tensorboard(self.writer, average_reward, episode)
