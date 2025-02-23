import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import torch
from config.constants import Constants as C
import boto3
import random
from typing import List
import psutil
import os
    

def convert_to_binary_tensor(data: list[list[int]], pass_option: bool = False) -> torch.Tensor:
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
    scalar_n_cards = list_to_change[position]
    del list_to_change[position]
    list_to_change.append(scalar_n_cards)

    return list_to_change


def shuffle_list(input_list):
    "Randomly reorders the elements in a list."
    shuffled = input_list.copy()
    random.shuffle(shuffled)
    return shuffled


def log_vram_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"MEMORY [{tag}] - Allocated: {allocated:.4f}MB, Reserved: {reserved:.4f}MB")


def log_ram_memory(tag=""):
    """
    Log current RAM (system memory) usage with an optional tag to identify where it's called from.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    rss_mb = memory_info.rss / (1024 * 1024)  # Resident Set Size
    vms_mb = memory_info.vms / (1024 * 1024)  # Virtual Memory Size
    
    system_memory = psutil.virtual_memory()
    system_used_percent = system_memory.percent
    system_used_mb = system_memory.used / (1024 * 1024)
    system_total_mb = system_memory.total / (1024 * 1024)
    
    print(f"RAM MEMORY [{tag}]:")
    print(f"  Process RSS: {rss_mb:.2f}MB (actual memory used by process)")
    print(f"  Process VMS: {vms_mb:.2f}MB (virtual memory allocated)")
    print(f"  System Usage: {system_used_mb:.0f}MB / {system_total_mb:.0f}MB ({system_used_percent}%)")


def take_n_highest_tensor(tensor: torch.Tensor, position: int=2) -> float:
    "Starts in 1"
    sorted_vals, _ = torch.sort(tensor, dim=1, descending=True)
    return sorted_vals[:, position-1].mean().item()


def take_median_tensor(tensor: torch.Tensor, position: int=2) -> float:
    sorted_vals, _ = torch.sort(tensor, dim=1, descending=True)
    return sorted_vals[:, position-1].mean().item()




if __name__ == "__main__":
    upload_to_s3()