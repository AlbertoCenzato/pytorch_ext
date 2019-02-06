from __future__ import annotations

import abc
from typing import Callable, Optional, List

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import visdom_board


def call_all(callables: List[Callable]) -> None:
    for f in callables:
        f()


class TrainingCallback:

    def __init__(self):
        self.trainer = None

    @abc.abstractmethod
    def __call__(self) -> None:
        raise NotImplementedError()


class BatchTrainingCallback:

    def __init__(self):
        self.trainer = None

    @abc.abstractmethod
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


# TODO: ModelTrainerParams does not contain batch_size!!!
class ModelTrainerParams:

    def __init__(self, epochs):
        self.epochs = epochs


class ModelTrainer:
    """
        Model training class.
        Given a model, a dataset and training parameters ModelTrainer.run() 
        takes care of training the model. The user should use 
        ModelTrainer.add_batch_training_callback() to provide the detalis on 
        how to feed data into the model and how to compute the batch loss. 
        See method documentation for details.

        Fileds accessible from TrainingCallback:
            loss_fn: Callable
            optimizer: torch.optim.Optimizer
            epochs: int
            current_epoch: int
            device: torch.device
            vm: visdom_board.VisdomManager
            model: torch.nn.Module
            training_data_loader: torch.utils.data.DataLoader
            validation_data_loader: torch.utils.data.DataLoader

        The trainer can be further specialized adding pre-training, post-training, 
        pre-epoch, post-epoch, pre-batch and post-batch callbacks.
    """

    def __init__(self, loss_fn: Callable, epochs: int, optimizer: Optimizer, 
                 device=torch.device('cpu')):
        """
            Args:
             @loss_fn: a callable object or a function. It can be a Pytorch loss function 
                       (e.g torch.nn.MSELoss or torch.nn.CrossEntropyLoss) or any other function
                       with the same signature.
             @epochs: training epochs
             @optimizer: a Pytorch optimizer; see torch.optim
             @device: device to use
        """
        self.loss_fn   = loss_fn
        self.optimizer = optimizer
        self.epochs    = epochs
        self.current_epoch = -1
        self.device    = device

        self.model = None
        self.training_data_loader   = None
        self.validation_data_loader = None

        # callbacks
        class StubBatchCallback(BatchTrainingCallback): 
            def __call__(self, batch: torch.Tensor):
                raise NotImplementedError('You should specify a batch training callback ' +
                                          'using ModelTrainer.add_batch_training_callback()')
        self.batch_training = None
        self.add_batch_training_callback(StubBatchCallback())
        self.pre_training_actions  = []
        self.pre_epoch_actions     = []
        self.pre_batch_actions     = []
        self.post_batch_actions    = []
        self.post_epoch_actions    = []
        self.post_training_actions = []

        # initialize VisdomBoard tools
        self.vm = visdom_board.get_visdom_manager()
        self.vm.close_all()

        with self.vm.environment('Training'):
            self._output_console = self.vm.get_output_console()
            self._loss_plot      = self.vm.get_line_plot(title='Training loss', 
                                                         xaxis='epochs',
                                                         yaxis='loss')  # ,
                                                         # opts=dict(title='Training loss',
                                                         #          xlabel='Epochs',
                                                         #          ylabel='Loss'))
        self.train_logging_period = 10

    def run(self, model: Module, training_set: DataLoader, training_callback: Optional[BatchTrainingCallback]=None, 
            validation_set: Optional[DataLoader] = None) -> None:
        """
            Runs a complete training on model. If validation_set is set to None evaluate_loss()
            will not be called.
        """
        if training_callback:
            self.add_batch_training_callback(training_callback)

        self.model = model
        self.training_data_loader   = training_set
        self.validation_data_loader = validation_set
        self.model.to(self.device)  # ensure model and data are on the same device

        call_all(self.pre_training_actions)
        
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            self.current_epoch = epoch
            call_all(self.pre_epoch_actions)
            self._train_epoch()
            call_all(self.post_epoch_actions)

        call_all(self.post_training_actions)
    
    def save_params(self) -> ModelTrainerParams:
        return ModelTrainerParams(self.epochs)

    def add_batch_training_callback(self, callback: BatchTrainingCallback) -> ModelTrainer:
        """ 
            Adds to ModelTrainer a TrainingCallback which only input argument 
            is the batch as given by the training data loader provided to ModelTrainer.
            The callback can access all ModelTrainer fields through self.trainer. 
            For example a common use case for a supervised labeling task could be:

            class ExampleSupervisedBatchTrainer(TrainingCallback):
                def __call__(self, batch) -> torch.Tensor:
                    data, labels = batch
                    output = self.trainer.model(data)
                    return self.trainer.loss_fn(output, labels)

            The callback must return a torch.Tensor representing the loss for the given batch.
            ModelTrainer will then take care of the backpropagation an weights update.
        """
        callback.trainer = self
        self.batch_training = callback
        return self

    def add_pre_training_action(self, callback: TrainingCallback) -> ModelTrainer:
        callback.trainer = self
        self.pre_training_actions.append(callback)
        return self

    def add_post_training_action(self, callback: TrainingCallback) -> ModelTrainer:
        callback.trainer = self
        self.post_training_actions.append(callback)
        return self

    def add_pre_epoch_action(self, callback: TrainingCallback) -> ModelTrainer:
        callback.trainer = self
        self.pre_epoch_actions.append(callback)
        return self

    def add_post_epoch_action(self, callback: TrainingCallback) -> ModelTrainer:
        callback.trainer = self
        self.post_epoch_actions.append(callback)
        return self

    def add_pre_batch_action(self, callback: TrainingCallback) -> ModelTrainer:
        callback.trainer = self
        self.pre_batch_actions.append(callback)
        return self

    def add_post_batch_action(self, callback: TrainingCallback) -> ModelTrainer:
        callback.trainer = self
        self.post_batch_actions.append(callback)
        return self

    # ----- private functions -----

    def _train_epoch(self) -> None:
        training_data_len = len(self.training_data_loader)
        self.running_loss = 0.0
        for batch_index, data in enumerate(self.training_data_loader, 0):
            data = data.to(self.device)
            
            call_all(self.pre_batch_actions)

            self.optimizer.zero_grad()  # reset parameters gradient
            batch_loss = self.batch_training(data)
            batch_loss.backward()        
            self.optimizer.step()

            self.running_loss += batch_loss.item()
            call_all(self.post_batch_actions)

            # print statistics
            if batch_index % self.train_logging_period == self.train_logging_period-1:  # print every time one-tenth of the dataset has been processed
                mean_loss = self.running_loss / self.train_logging_period  # print mean loss over the processed batches
                self._output_console.print('[epoch: {:.0f}, batch: {:.0f}] - loss: {:.3f}'
                                           .format(self.current_epoch + 1, batch_index + 1, mean_loss))
                self._loss_plot.append([self.current_epoch + batch_index/training_data_len], [mean_loss])
                self.running_loss = 0.0
