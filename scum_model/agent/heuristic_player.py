import torch
import sys
import os
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def heuristic_play(possible_actions: np.ndarray, state: torch.Tensor):
    conservative_play = np.min(possible_actions)
    peaky_possibility = np.sum(possible_actions % 14 == 5) > 0
    peaky_play = np.max(possible_actions[possible_actions % 14 == 5]) if peaky_possibility else 0
    possible_actions[-1] = - possible_actions[-1] ## avoid passing
    agressive_play = np.max(possible_actions) 
    ratio_of_difference = extract_ratio_btw_player_and_opp(state)

    if ratio_of_difference < 0.6:
        final_play = agressive_play
    elif (extract_number_of_cards_player(state) < 5) and peaky_possibility:
        final_play = peaky_play
    else:
        final_play = conservative_play
    return final_play


def extract_number_of_cards_player(state: torch.Tensor):
    return (state[77]*11)

def extract_ratio_btw_player_and_opp(state: torch.Tensor):
    opponent_cards = state[72:76]
    own_n_cards = state[77]
    min_number_of_cards = np.min(opponent_cards.clone().detach().cpu().numpy())
    return (min_number_of_cards*11) / (own_n_cards*11)


