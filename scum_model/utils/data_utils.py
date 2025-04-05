import torch
from torch.distributions import Categorical
from typing import Generator, Tuple, List

from config.constants import Constants as C

def shuffle_data(data):
    states, returns, action_masks, actions, old_log_probs, next_actions = data
    num_transitions = len(states)

    # Generate a permutation index
    permutation = torch.randperm(num_transitions)

    # Apply the permutation to shuffle the data
    shuffled_states = states[permutation]
    shuffled_rewards = returns[permutation]
    shuffled_action_masks = action_masks[permutation]
    shuffled_actions = actions[permutation]
    shuffled_log_probs = old_log_probs[permutation]
    shuffled_next_actions = next_actions[permutation]

    return shuffled_states, shuffled_rewards, shuffled_action_masks, shuffled_actions, shuffled_log_probs, shuffled_next_actions


def remove_impossible_states(data):
    states, rewards, action_masks, actions, old_log_probs, next_actions = data
    impossible_states = ~torch.all(states == 0, dim=1)
    states = states[impossible_states]
    rewards = rewards[impossible_states]
    action_masks = action_masks[impossible_states]
    old_log_probs = old_log_probs[impossible_states]
    next_actions = next_actions[impossible_states]

    return states, rewards, action_masks, actions, old_log_probs, next_actions


def create_batches(data, batch_size: int):
    states, returns, action_space, actions, old_log_probs, next_actions = data
    num_batches = (len(states) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, len(states))
        yield states[start:end], returns[start:end], action_space[start:end], actions[start:end], old_log_probs[start:end], next_actions[start:end]


def mask_impossible_actions(action_masks: torch.Tensor, policy_logits: torch.Tensor) -> torch.Tensor:
    masked_policy_logits = policy_logits.clone()
    masked_policy_logits[(action_masks==0)] = float('-inf')
    return masked_policy_logits

def compute_advantadge(returns: torch.Tensor, value_preds: torch.Tensor) -> torch.Tensor:
    return returns - value_preds


def compute_discounted_returns(rewards: list, discount: float) -> torch.Tensor:
    G = 0
    returns = []
    for reward in rewards[::-1]:
        G = reward + discount * G
        returns.insert(0, G)
    return torch.tensor(returns, device=C.DEVICE)


def create_only_pass_state(number_players: int) -> torch.Tensor:
    generated_data = [0 for _ in range((C.NUMBER_OF_POSSIBLE_STATES - 1) + number_players + C.NUMBER_OF_CARDS_PER_SUIT+1)] + [1]
    return torch.tensor(generated_data, dtype=torch.float32).to(C.DEVICE)


def compact_form_of_states(states: torch.Tensor) -> torch.Tensor:
    cards = states[:C.NUMBER_OF_POSSIBLE_STATES - 1]
    cards_matrix = cards.view(4, 14)
    compact_cards = cards_matrix.sum(dim=0)/C.NUMBER_OF_SUITS

    compact_cards[C.NUMBER_OF_CARDS_PER_SUIT] = torch.where(compact_cards[C.NUMBER_OF_CARDS_PER_SUIT] > 0, 1, 0)
    compact_state = torch.cat((compact_cards,states[(C.NUMBER_OF_POSSIBLE_STATES - 1):]))

    return compact_state


def is_action_valid(self, prediction, action_space):
    action_to_take = Categorical(prediction).sample()
    valid = action_space[action_to_take] != 0
    return C.REWARD_CHOOSE_IMPOSIBLE_ACTION if valid == False else 0

