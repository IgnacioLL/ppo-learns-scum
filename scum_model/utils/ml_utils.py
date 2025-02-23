import torch 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from config.constants import Constants as C

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
        if self.last_epoch >= self.warmup_steps:
            return self.target_lr
        else:
            return self.initial_lr + self.lr_step * self.last_epoch
        
    

class CoolerLRScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup period.
    Gradually increases learning rate from initial_lr to target_lr over warmup_steps,
    then maintains target_lr for remaining steps.
    """
    def __init__(self, optimizer, cooler_steps, initial_lr=1e-4, target_lr=1e-5, last_epoch=-1):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            initial_lr: Initial learning rate at start of warmup
            target_lr: Target learning rate after warmup
            last_epoch: The index of last epoch
        """
        self.cooler_steps = cooler_steps
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.lr_step = (target_lr - initial_lr) / cooler_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate based on current step."""
        if self.last_epoch >= self.cooler_steps:
            return [self.target_lr for _ in self.base_lrs]
        
        current_lr = self.initial_lr + self.lr_step * self.last_epoch
        return [current_lr for _ in self.base_lrs]
    

    def get_current_learning_rate(self):
        if self.last_epoch >= self.cooler_steps:
            return self.target_lr
        else:
            return self.initial_lr + self.lr_step * self.last_epoch