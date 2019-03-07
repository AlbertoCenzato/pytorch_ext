from typing import Callable
import os

import torch
from torch.optim import Optimizer

from .model_trainer import ModelTrainer
from .callbacks import TrainingTimeEstimation, BatchStatistics, Checkpoint


def create_model_trainer(loss_fn: Callable, epochs: int, optimizer: Optimizer,
                         device=torch.device('cpu'), checkpoint_dir=None) -> ModelTrainer:
    """
    Creates a ModelTrainer with the most common MTCallbacks already added.
    It has TrainingTimeEstimation, BatchStatics and Checkpoint already attached
    :param loss_fn:
    :param epochs:
    :param optimizer:
    :param device:
    :param checkpoint_dir:
    :return: a ModelTrainer instance
    """
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
    model_trainer = ModelTrainer(loss_fn, epochs, optimizer, device)
    model_trainer.attach_callback(TrainingTimeEstimation()) \
                 .attach_callback(BatchStatistics(10)) \
                 .attach_callback(Checkpoint(checkpoint_dir, 5))

    return model_trainer
