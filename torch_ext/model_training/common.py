from typing import Callable
import os

import torch
from torch.optim import Optimizer

from .model_trainer import ModelTrainer
from .callbacks import TrainingTimeEstimation, BatchStatistics, Checkpoint  # ProgressiveNetInspector


def create_model_trainer(loss_fn: Callable, epochs: int, optimizer: Optimizer,
                         device=torch.device('cpu'), summary_period=10, checkpoint_dir=None,
                         checkpoint_period=5) -> ModelTrainer:
    """
    Creates a ModelTrainer with the most common MTCallbacks already added.
    It has TrainingTimeEstimation, BatchStatics and Checkpoint already attached
    :param loss_fn:
    :param epochs:
    :param optimizer:
    :param device:
    :param summary_period:
    :param checkpoint_dir:
    :param checkpoint_period:
    :return: a ModelTrainer instance
    """
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
    model_trainer = ModelTrainer(loss_fn, epochs, optimizer, device)
    model_trainer.attach_callback(TrainingTimeEstimation()) \
                 .attach_callback(BatchStatistics(summary_period)) \
                 .attach_callback(Checkpoint(checkpoint_dir, checkpoint_period))

    return model_trainer
