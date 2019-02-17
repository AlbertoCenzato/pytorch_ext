from typing import Callable

import torch
from torch.optim import Optimizer

from .model_trainer import ModelTrainer
from .callbacks import TrainingTimeEstimation, BatchStatistics


def create_model_trainer(loss_fn: Callable, epochs: int, optimizer: Optimizer,
                         device=torch.device('cpu')) -> ModelTrainer:
    """
    Creates a ModelTrainer with the most common TrainingCallbacks already added.
    It has TrainingTimeEstimation, BatchStatics already attached
    :param loss_fn:
    :param epochs:
    :param optimizer:
    :param device:
    :return:
    """
    model_trainer = ModelTrainer(loss_fn, epochs, optimizer, device)
    model_trainer.add_post_epoch_action(TrainingTimeEstimation()) \
                 .add_post_batch_action(BatchStatistics(10))

    return model_trainer
