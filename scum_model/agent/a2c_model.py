from config.constants import Constants as C
import torch.nn as nn


class A2CModel(nn.Module):
    def __init__(self, number_of_players):
        super().__init__()

        self.chore_part = nn.Sequential(
            nn.Linear(C.NUMBER_OF_POSSIBLE_STATES + C.NUMBER_OF_CARDS_PER_SUIT + 1 + number_of_players, 256),
            nn.LayerNorm(256),  # Use LayerNorm instead of BatchNorm1d
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),  # Use LayerNorm instead of BatchNorm1d
            nn.ReLU()
        )

        self.value_estimate = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # Use LayerNorm instead of BatchNorm1d
            nn.ReLU(), 
            nn.Linear(256, 1)
        )
        
        self.action_probability = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # Use LayerNorm instead of BatchNorm1d
            nn.ReLU(), 
            nn.Linear(256, C.NUMBER_OF_POSSIBLE_STATES)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if x.dim() == 1:  # If input is 1D, reshape to 2D by adding batch dimension
            x = x.unsqueeze(0)
    
        x = self.chore_part(x)
        value = self.value_estimate(x)
        action_probs = self.softmax(self.action_probability(x))
        return value, action_probs
