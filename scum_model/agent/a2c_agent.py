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
import pandas as pd

import torch.nn as nn

from nnet.nnet import NNet

from utils import data_utils, logging, loss_utils, utils
from utils.ml_utils import CoolerLRScheduler
from buffer.buffer import Buffer

from typing import Union

class A2CAgent:
    def __init__(self, training: bool=False, learning_rate: float = 1e-5, discount: float = None, number_players: int = C.NUMBER_OF_AGENTS, path: str = None, model="big", model_id=None, entropy_coef=0, policy_error_coef=1, value_error_coef=1, epochs=C.N_EPOCH_PER_STEP):
        self.model = NNet(number_of_players=number_players, model=model, id=model_id).to(C.DEVICE)
        if path:
            self.model.load_state_dict(torch.load(path, weights_only=False))

        self.learning_rate = learning_rate
        self.initial_learning_rate = self.learning_rate * C.INITIAL_LR_FACTOR
        self.current_learning_rate = self.initial_learning_rate
        self.discount = discount if discount else C.DISCOUNT

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = CoolerLRScheduler(
            self.optimizer,
            cooler_steps=C.COOLER_STEPS,
            initial_lr=learning_rate*C.INITIAL_LR_FACTOR,
            target_lr=learning_rate
        )
        self.eps = np.finfo(np.float32).eps.item()

        self.policy_error_coef = policy_error_coef
        self.value_error_coef = value_error_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.training = training
        self.buffer = Buffer()

        self.episode = None

    def set_training(self, set_to_train: bool) -> None:
        self.training = set_to_train

    def decide_move(self, state: torch.Tensor, action_space: torch.Tensor) -> Union[int, torch.Tensor]:
        if state is None:
            state = data_utils.create_only_pass_state()
        else:
            compact_state = data_utils.compact_form_of_states(state)
            prediction = self.predict(compact_state)

            action_space = action_space.unsqueeze(0)
            masked_predictions = data_utils.mask_impossible_actions(action_space, prediction)
            masked_predictions_prob = F.softmax(masked_predictions, dim=-1)
            logging.log_current_state_and_prediction(state, prediction)

            prediction_masked = Categorical(masked_predictions_prob)
            action = prediction_masked.sample()
            
            log_prob = prediction_masked.log_prob(action)
            return action.item() + 1, log_prob    ## esto devolvera un valor entre 1 y 57 que sera la eleccion del modelo

    @torch.no_grad()
    def predict(self, state):
        _, probs = self.model.forward(state)
        return probs.detach().clone()  # Explicitly detach and clone


    def train(self, step, batch_size=32):
        self.step = step
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
        
        return logging.average_metrics(metrics)


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

        ratio_5th_epoch = torch.abs(ratio - 1).mean().item() if self.epoch == 5 else None
        ratio_7th_epoch = torch.abs(ratio - 1).mean().item() if self.epoch == 6 else None
        ratio_10th_epoch = torch.abs(ratio - 1).mean().item() if self.epoch == 9 else None

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
            'ratio_7th_epoch': ratio_7th_epoch,
            'ratio_10th_epoch': ratio_10th_epoch,
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

