import torch
import pandas as pd
from utils import data_utils
import gc
from config.constants import Constants as C

from typing import List

class Buffer:
    def __init__(self):
        self.buffer = {'states': [], 'rewards': [], 'returns': [], 'action_space': [], 'actions': [], 'old_log_probs': []}

    def save_in_buffer(self, current_state, reward, action_space, action, log_prob):
        # Detach any tensors that might have gradients
        if isinstance(current_state, torch.Tensor):
            current_state = current_state.detach().cpu()
        if isinstance(action_space, torch.Tensor):
            action_space = action_space.detach().cpu()
        if isinstance(log_prob, torch.Tensor):
            log_prob = log_prob.detach().cpu().item()
        
        compact_state = data_utils.compact_form_of_states(current_state)
        self.buffer['states'].append(compact_state)
        self.buffer['rewards'].append(reward)
        self.buffer['action_space'].append(action_space)
        self.buffer['actions'].append(action)
        self.buffer['old_log_probs'].append(log_prob)

    def flush_buffer(self, path):
        data = {k: self.buffer[k] for k in ('actions', 'rewards', 'old_log_probs', 'returns')}
        df = pd.DataFrame.from_dict(data)
        df.to_parquet(path)

    def load_data_from_buffer(self):
        states = torch.stack([state.to(C.DEVICE) for state in self.buffer['states']])
        returns = torch.tensor([reward for reward in self.buffer['returns']], device=C.DEVICE)
        action_masks = torch.stack([action_space.to(C.DEVICE) for action_space in self.buffer['action_space']])
        actions = torch.tensor([action for action in self.buffer['actions']], device=C.DEVICE)
        old_log_probs = torch.tensor([log_prob for log_prob in self.buffer['old_log_probs']], device=C.DEVICE)
        return states, returns, action_masks, actions, old_log_probs
    
    def clear_buffer(self):
        self.buffer = {'states': [], 'rewards': [], 'returns': [], 'action_space': [], 'actions': [], 'old_log_probs': []}
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def apply_discounted_returns_in_buffer(self, agent_rewards: List[float], discount: float):
        returns: torch.Tensor = data_utils.compute_discounted_returns(agent_rewards, discount)
        self.buffer['returns'] = self.buffer['returns'] + returns.cpu().numpy().tolist()
