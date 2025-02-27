import sys
import os
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from config.constants import Constants as C
import torch
import torch.nn as nn
import uuid

from typing import Tuple

TWO_OF_HEARTS = 1
PASS_ACTION = 1
THROWN_CARDS_FEATURES = (C.NUMBER_OF_CARDS_PER_SUIT + 1)
OTHER_PLAYERS_FEATURES = C.NUMBER_OF_AGENTS
ACTION_SPACE = (C.NUMBER_OF_CARDS_PER_SUIT + TWO_OF_HEARTS + PASS_ACTION)
DIM_INPUT = (C.NUMBER_OF_CARDS_PER_SUIT + TWO_OF_HEARTS) + PASS_ACTION  + ACTION_SPACE + THROWN_CARDS_FEATURES + OTHER_PLAYERS_FEATURES 

def hu_initialization(tensor):
    if isinstance(tensor, nn.Linear):
        fan_in = tensor.in_features
        std = math.sqrt(2.0 / fan_in)
        nn.init.normal_(tensor.weight, mean=0, std=std)  # Random weight initialization
        
        if tensor.bias is not None:
            # Random bias initialization
            nn.init.normal_(tensor.bias, mean=0, std=std)  # Use a normal distribution for bias as well


class NNet(nn.Module):
    def __init__(self, number_of_players, model="big", id=None):
        super().__init__()
        self.number_of_players = number_of_players
        self.size = model
        self.id = str(uuid.uuid4()) if id is None else id
        self.chore_part = False if model == 'large_sep_arch' else True
        if model == "large_sep_arch":
            self.create_separate_arch_model(neurons=1024)
        
        if model == "large":
            self.create_model(neurons=1024)
        
        if model == "big":
            self.create_model(neurons=256)

        elif model == "medium":
            self.create_model(neurons=128)

        elif model == "small":
            self.create_small_model(neurons=64)
        
        self.apply(hu_initialization)

    def create_separate_arch_model(self, neurons: int):
        self.value_estimate = nn.Sequential(
            nn.Linear(DIM_INPUT, neurons),
            nn.LayerNorm(neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons*2),
            nn.LayerNorm(neurons*2),
            nn.LeakyReLU(),
            nn.Linear(neurons*2, neurons),
            nn.LayerNorm(neurons),
            nn.LeakyReLU(), 
            nn.Linear(neurons, neurons//4),
            nn.LeakyReLU(), 
            nn.Linear(neurons//4, 1)
        )
        
        self.policy_probability = nn.Sequential(
            nn.Linear(DIM_INPUT, neurons),
            nn.LayerNorm(neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons*2),
            nn.LayerNorm(neurons*2),
            nn.LeakyReLU(),
            nn.Linear(neurons*2, neurons),
            nn.LayerNorm(neurons),
            nn.LeakyReLU(), 
            nn.Linear(neurons, C.NUMBER_OF_POSSIBLE_STATES)
        )


    def create_model(self, neurons: int):
        self.chore_part = nn.Sequential(
            nn.Linear(DIM_INPUT, neurons),
            nn.LayerNorm(neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons*2),
            nn.LayerNorm(neurons*2),
            nn.ReLU()
        )

        self.value_estimate = nn.Sequential(
            nn.Linear(neurons*2, neurons),
            nn.LayerNorm(neurons),
            nn.ReLU(), 
            nn.Linear(neurons, 1)
        )
        
        self.policy_probability = nn.Sequential(
            nn.Linear(neurons*2, neurons),
            nn.LayerNorm(neurons),
            nn.ReLU(), 
            nn.Linear(neurons, C.NUMBER_OF_POSSIBLE_STATES)
        )

    def create_small_model(self):
        self.chore_part = nn.Sequential(
            nn.Linear(DIM_INPUT, 64),
            nn.ReLU()
        )

        self.value_estimate = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(), 
            nn.Linear(64, 1)
        )
        
        self.policy_probability = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(), 
            nn.Linear(128, C.NUMBER_OF_POSSIBLE_STATES)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:  # If input is 1D, reshape to 2D by adding batch dimension
            x = x.unsqueeze(0)

        if self.chore_part:
            x = self.chore_part(x)
        value = self.value_estimate(x)
        action_logits = self.policy_probability(x)

        return value, action_logits