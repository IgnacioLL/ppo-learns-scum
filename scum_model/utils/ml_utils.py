import torch 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from config.constants import Constants as C

def compute_grad_stats(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:  # Check if the gradient is computed
            grads.append(param.grad.view(-1))  # Flatten the gradient tensor

    if grads:
        all_grads = torch.cat(grads)  # Concatenate all gradients into one tensor
        median_grad = all_grads.median()
        mean_grad = all_grads.mean()
        max_grad = all_grads.max()
        min_grad = all_grads.min()
        p99_grad = torch.quantile(all_grads, 0.99)
        p01_grad = torch.quantile(all_grads, 0.01)
        return median_grad, mean_grad, max_grad, min_grad, p99_grad, p01_grad
    else:
        return None, None  # No gradients present (e.g., in untrained parameters)
    
def compute_discounted_returns(rewards: list, discount) -> torch.tensor:
    G = 0
    returns = []
    for reward in rewards[::-1]:
        G = reward + discount * G
        returns.insert(0, G)
    return torch.tensor(returns, device=C.DEVICE)


import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupLRScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup period.
    Gradually increases learning rate from initial_lr to target_lr over warmup_steps,
    then maintains target_lr for remaining steps.
    """
    def __init__(self, optimizer, warmup_steps, initial_lr=1e-6, target_lr=1e-3, last_epoch=-1):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            initial_lr: Initial learning rate at start of warmup
            target_lr: Target learning rate after warmup
            last_epoch: The index of last epoch
        """
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.lr_step = (target_lr - initial_lr) / warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate based on current step."""
        if self.last_epoch >= self.warmup_steps:
            return [self.target_lr for _ in self.base_lrs]
        
        current_lr = self.initial_lr + self.lr_step * self.last_epoch
        return [current_lr for _ in self.base_lrs]
    

    def get_current_learning_rate(self):
        return self.initial_lr + self.lr_step * self.last_epoch