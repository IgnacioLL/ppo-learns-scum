import sys
import os
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from config.constants import Constants as C
import torch.nn as nn
import uuid


def hu_initialization(tensor):
    if isinstance(tensor, nn.Linear):
        fan_in = tensor.in_features
        std = math.sqrt(2.0 / fan_in)
        nn.init.normal_(tensor.weight, mean=0, std=std)  # Random weight initialization
        
        if tensor.bias is not None:
            # Random bias initialization
            nn.init.normal_(tensor.bias, mean=0, std=std)  # Use a normal distribution for bias as well


class A2CModel(nn.Module):
    def __init__(self, number_of_players, model="big", id=None):
        super().__init__()
        self.number_of_players = number_of_players
        self.size = model
        self.id = str(uuid.uuid4()) if id is None else id

        if model == "big":
            self.create_big_model()

        elif model == "medium":
            self.create_medium_model()

        elif model == "small":
            self.create_small_model()
        
        self.softmax = nn.Softmax(dim=-1)
        self.apply(hu_initialization)

    def create_big_model(self):
        self.chore_part = nn.Sequential(
            nn.Linear(C.NUMBER_OF_POSSIBLE_STATES + C.NUMBER_OF_CARDS_PER_SUIT + 1 + self.number_of_players, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )

        self.value_estimate = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(), 
            nn.Linear(256, 1)
        )
        
        self.policy_probability = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(), 
            nn.Linear(256, C.NUMBER_OF_POSSIBLE_STATES)
        )

    def create_medium_model(self):
        self.chore_part = nn.Sequential(
            nn.Linear(C.NUMBER_OF_POSSIBLE_STATES + C.NUMBER_OF_CARDS_PER_SUIT + 1 + self.number_of_players, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.value_estimate = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(), 
            nn.Linear(128, 1)
        )
        
        self.policy_probability = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(), 
            nn.Linear(128, C.NUMBER_OF_POSSIBLE_STATES)
        )

    def create_small_model(self):
        self.chore_part = nn.Sequential(
            nn.Linear(C.NUMBER_OF_POSSIBLE_STATES + C.NUMBER_OF_CARDS_PER_SUIT + 1 + self.number_of_players, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        self.value_estimate = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(), 
            nn.Linear(64, 1)
        )
        
        self.policy_probability = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(), 
            nn.Linear(128, C.NUMBER_OF_POSSIBLE_STATES)
        )

    def forward(self, x):
        if x.dim() == 1:  # If input is 1D, reshape to 2D by adding batch dimension
            x = x.unsqueeze(0)
    
        x = self.chore_part(x)
        value = self.value_estimate(x)
        action_probs = self.softmax(self.policy_probability(x))

        return value, action_probs