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
LAST_ACTION = 1
DIM_INPUT = (C.NUMBER_OF_CARDS_PER_SUIT + TWO_OF_HEARTS) + PASS_ACTION  + ACTION_SPACE + THROWN_CARDS_FEATURES + OTHER_PLAYERS_FEATURES + LAST_ACTION

LATENT_SPACE_NEXT_ACTION = 32
EMBEDDING_SIZE_OUTPUT = 16

HELPER_DIM_INPUT = DIM_INPUT - 1 + EMBEDDING_SIZE_OUTPUT 
FINAL_DIM_INPUT = DIM_INPUT - 1 + EMBEDDING_SIZE_OUTPUT + LATENT_SPACE_NEXT_ACTION

def hu_initialization(tensor):
    if isinstance(tensor, nn.Linear):
        fan_in = tensor.in_features
        std = math.sqrt(2.0 / fan_in)
        nn.init.normal_(tensor.weight, mean=0, std=std)  # Random weight initialization
        
        if tensor.bias is not None:
            # Random bias initialization
            nn.init.normal_(tensor.bias, mean=0, std=std)  # Use a normal distribution for bias as well



class NNet(nn.Module):
    def __init__(self, number_of_players, model, model_id):
        super().__init__()
        self.number_of_players = number_of_players
        self.size = model
        self.model_id = model_id
        self.chore_part = False if str(model).endswith('-sep-arch') else True

        self.create_embedding()

        if model == "large-sep-arch":
            self.create_separate_arch_model(neurons=1024)
        elif model == "big-sep-arch":
            self.create_separate_arch_model(neurons=256)
        elif model == "medium-sep-arch":
            self.create_separate_arch_model(neurons=128)
        elif model == "small-sep-arch":
            self.create_separate_arch_model(neurons=64)
        elif model == "large":
            self.create_model(neurons=1024)
        elif model == "big":
            self.create_model(neurons=256)
        elif model == "medium":
            self.create_model(neurons=128)
        elif model == 'small':
            self.create_small_model()
        
        self.apply(hu_initialization)

    def create_separate_arch_model(self, neurons: int):
        self.value_estimate = nn.Sequential(
            nn.Linear(FINAL_DIM_INPUT, neurons),
            nn.LayerNorm(neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons*2),
            nn.LayerNorm(neurons*2),
            nn.LeakyReLU(),
            nn.Linear(neurons*2, neurons),
            nn.LayerNorm(neurons),
            nn.LeakyReLU(), 
            nn.Linear(neurons, neurons // 4),
            nn.LeakyReLU(), 
            nn.Linear(neurons // 4, 1)
        )
        
        self.policy_probability = nn.Sequential(
            nn.Linear(FINAL_DIM_INPUT, neurons),
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
            nn.Linear(FINAL_DIM_INPUT, neurons),
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
            nn.Linear(FINAL_DIM_INPUT, 64),
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

    def create_embedding(self):
        self.embedding = nn.Sequential(
            nn.Embedding(C.NUMBER_OF_POSSIBLE_STATES + 2, EMBEDDING_SIZE_OUTPUT),
            nn.LayerNorm(EMBEDDING_SIZE_OUTPUT)
        )


    def forward(self, x: torch.Tensor, latent_space: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:  # If input is 1D, reshape to 2D by adding batch dimension
            x = x.unsqueeze(0)

        state = x[:, :DIM_INPUT - 1]
        last_action = x[:, DIM_INPUT - 1].to(torch.long) + 1

        embedded_last_action = self.embedding(last_action)

        x_and_latent_space = torch.cat([state, embedded_last_action, latent_space], dim=1)
        if self.chore_part:
            x_and_latent_space = self.chore_part(x_and_latent_space)
        x_and_latent_space = x_and_latent_space.to(dtype=torch.float32)
        value = self.value_estimate(x_and_latent_space)
        action_logits = self.policy_probability(x_and_latent_space)
        return value, action_logits


class HelperNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.create_next_action_model()
        self.create_embedding()


    def create_embedding(self):
        self.embedding = nn.Sequential(
            nn.Embedding(C.NUMBER_OF_POSSIBLE_STATES + 2, EMBEDDING_SIZE_OUTPUT),
            nn.LayerNorm(EMBEDDING_SIZE_OUTPUT)
        )


    def create_next_action_model(self, neurons: int=128, output_dim: int=LATENT_SPACE_NEXT_ACTION) -> None:
        self.next_action_prediction_chore = nn.Sequential(
            nn.Linear(HELPER_DIM_INPUT, neurons),
            nn.LayerNorm(neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons*2),
            nn.LayerNorm(neurons*2),
            nn.ReLU(),
            nn.Linear(neurons*2, neurons),
            nn.LayerNorm(neurons),
            nn.ReLU(), 
            nn.Linear(neurons, output_dim),
        )

        self.next_action_player1 = self.head_next_action(output_dim)
        self.next_action_player2 = self.head_next_action(output_dim)
        self.next_action_player3 = self.head_next_action(output_dim)
        self.next_action_player4 = self.head_next_action(output_dim)


    def head_next_action(self, output_dim):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(output_dim, C.NUMBER_OF_POSSIBLE_STATES + 1),
            nn.Softmax()        
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        if x.dim() == 1:  # If input is 1D, reshape to 2D by adding batch dimension
            x = x.unsqueeze(0)

        state = x[:, :DIM_INPUT - 1]
        last_action = x[:, DIM_INPUT - 1].to(torch.long) + 1

        embedded_last_action = self.embedding(last_action)

        x = torch.cat([state, embedded_last_action], dim=1)

        x = x.to(dtype=torch.float32)

        next_action_latent_space = self.next_action_prediction_chore(x)

        player1_next_action = self.next_action_player1(next_action_latent_space)
        player2_next_action = self.next_action_player2(next_action_latent_space)
        player3_next_action = self.next_action_player3(next_action_latent_space)
        player4_next_action = self.next_action_player4(next_action_latent_space)

        return next_action_latent_space, (player1_next_action, player2_next_action, player3_next_action, player4_next_action)
