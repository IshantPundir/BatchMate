"""
Early stops the training if validation loss doesn't improve after a given patience.
"""
import logging
from copy import deepcopy
from typing import Any, Callable

import numpy as np
from torch.nn import Module
from torch.optim.optimizer import Optimizer

log = logging.getLogger(__name__)

class BestState:
    def __init__(self, model:Module, optimizer:Optimizer, loss:float) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

class StopLoss:
    """
    Stop the training if validation loss doesn't improve after a given patience is exceded.
    """
    def __init__(self,patience:int=7, delta:float=0.0,  verbose:bool=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.delta = delta
        self.verbose = verbose
        self.patience = patience

        self.counter = 0
        self.val_loss_min = np.inf
        self.best_loss = None
        self.best_score = None
        self.best_model = None
        self.best_optimizer = None
        
    def __call__(self, model:Module, optimizer:Optimizer, val_loss:float) -> bool:
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.store_best_model(val_loss, model, optimizer)
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            log.warn(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.store_best_model(val_loss, model, optimizer)
            self.counter = 0

        return False
        
    def store_best_model(self, val_loss:float, model:Module, optimizer:Optimizer):
        if self.verbose:
            log.warn(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        self.val_loss_min = val_loss
        self.best_model = deepcopy(model)
        self.best_optimizer = deepcopy(optimizer)
        self.best_loss = val_loss

    def get_best_state(self) -> BestState:
        return BestState(model=self.best_model,
                         optimizer=self.best_optimizer,
                         loss=self.best_loss)