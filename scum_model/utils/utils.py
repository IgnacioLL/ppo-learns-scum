import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import torch
from config.constants import Constants as C
import boto3
from typing import List

def convert_to_binary_tensor(data: list[list[int]], pass_option: bool = False) -> torch.tensor:
    result = [[0] * 14 for _ in range(4)]

    # Convert each subarray (after the first) to a set for O(1) membership checks
    subarray_sets = [set(subarray) for subarray in data]

    # Iterate over each element in the first subarray
    for j in range(4):
        for i in range(1,15):
            # Check in each subsequent set
            if i in subarray_sets[j]:
                result[j][i-1] = 1
    if pass_option:
        return torch.tensor(np.append(np.array(result).flatten(), 1)).float().to(C.DEVICE)
    else:
        return torch.tensor(np.append(np.array(result).flatten(), 0)).float().to(C.DEVICE)

def print_rl_variables(reward: int, new_observation: np.array, done: bool, epsilon: float) -> None:
        print(f"The reward is: {reward}")
        print(f"The new observation is: {new_observation}")
        print(f"The done is: {done}")
        print(f"The epsilon is: {epsilon}")

def download_from_s3() -> None:
    s3 = boto3.client('s3')
    for i in range(C.NUMBER_OF_AGENTS):
        s3.download_file("dqn-scum", f"models/agent_{i+1}.pt", f"models/checkpoints/agent_{i+1}.pt")

def upload_to_s3() -> None:
    s3 = boto3.client('s3')
    for i in range(C.NUMBER_OF_AGENTS):
        s3.upload_file(f"models/checkpoints/agent_{i+1}.pt", "dqn-scum", f"models/agent_{i+1}.pt")

def move_to_last_position(list: List, position: int) -> List:
    list_to_change = list.copy()
    scalar_position = list_to_change[position]
    del list_to_change[position]
    list_to_change.sort()
    list_to_change.append(scalar_position)

    return list_to_change

def compact_form_of_states(states: torch.tensor) -> torch.tensor:
    cards = states[:C.NUMBER_OF_POSSIBLE_STATES - 1]
    cards_matrix = cards.view(4, 14)
    compact_cards = cards_matrix.sum(dim=0)/C.NUMBER_OF_SUITS

    compact_cards[C.NUMBER_OF_CARDS_PER_SUIT] = torch.where(compact_cards[C.NUMBER_OF_CARDS_PER_SUIT] > 0, 1, 0)
    compact_state = torch.cat((compact_cards,states[(C.NUMBER_OF_POSSIBLE_STATES - 1):]))

    return compact_state


if __name__ == "__main__":
    upload_to_s3()