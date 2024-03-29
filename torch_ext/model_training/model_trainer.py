from __future__ import annotations

import abc
import enum
from typing import Callable, Optional, List

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def call_all(callables: List[Callable]) -> None:
    for f in callables:
        f()


class Event(enum.Enum):

    ON_TRAINING_BEGIN = enum.auto()
    ON_TRAINING_END   = enum.auto()
    ON_EPOCH_BEGIN = enum.auto()
    ON_EPOCH_END   = enum.auto()
    ON_BATCH_BEGIN = enum.auto()
    ON_BATCH_END   = enum.auto()
    ANY = enum.auto()


class MTCallback:

    def __init__(self):
        self.trainer = None
        self.event   = Event.ANY

    @abc.abstractmethod
    def __call__(self) -> None:
        raise NotImplementedError()

    def on_attach(self) -> None:
        """
        This method is called when the callback is attached to an
        instance of ModelTrainer. When on_attach() is called it is
        guaranteed that self.trainer is set to the ModelTrainer instance
        it has been attached to. This method can be used to initialize
        the Callback's fields that need to access ModelTrainer data.
        See BatchStatistics for an example.
        """
        pass


class TrainingCallback:

    def __init__(self):
        self.trainer = None

    @abc.abstractmethod
    def __call__(self, model: torch.nn.Module, batch: torch.Tensor, loss_fn: Callable) -> torch.Tensor:
        raise NotImplementedError()

    def on_attach(self):
        """
        This method is called when the callback is attached to an
        instance of ModelTrainer. When on_attach() is called it is
        guaranteed that self.trainer is set to the ModelTrainer instance
        it has been attached to. This method can be used to initialize
        the Callback's fields that need to access ModelTrainer data.
        See BatchStatistics for an example.
        """
        pass


class ModelTrainer:
    """
    Model training class.
    Given a model, a dataset and training parameters ModelTrainer.run()
    takes care of training the model. The user should use
    ModelTrainer.add_batch_training_callback() to provide the details on
    how to feed data into the model and how to compute the batch loss.
    See method documentation for details.

    Fileds accessible from TrainingCallback:
        loss_fn: Callable
        optimizer: torch.optim.Optimizer
        epochs: int
        current_epoch: int
        current_batch: int
        last_batch_loss: float
        device: torch.device
        model: torch.nn.Module
        data_loader_tr: torch.utils.data.DataLoader
        data_loader_va: torch.utils.data.DataLoader

    The trainer can be further specialized adding pre-training, post-training,
    pre-epoch, post-epoch, pre-batch and post-batch callbacks.
    """

    def __init__(self, loss_fn: Callable, epochs: int, optimizer: Optimizer, 
                 device=torch.device('cpu')):
        """
        :param loss_fn: a callable object or a function. It can be a Pytorch loss function
                       (e.g torch.nn.MSELoss or torch.nn.CrossEntropyLoss) or any other function
                       with the same signature.
        :param epochs: training epochs
        :param optimizer: a Pytorch optimizer; see torch.optim
        :param device: device to use
        """
        self.loss_fn   = loss_fn
        self.optimizer = optimizer
        self.epochs    = epochs
        self.current_epoch = -1
        self.current_batch = -1
        self.last_batch_loss = 0.0
        self.device = device

        self.model = None
        self.data_loader_tr = None
        self.data_loader_va = None

        # callbacks
        class StubCallback(TrainingCallback):
            def __call__(self, model, batch, loss_fn):
                raise NotImplementedError('You should specify a batch training callback ' +
                                          'using ModelTrainer.add_batch_training_callback()')
        self._batch_training = None
        self.add_batch_training_callback(StubCallback())
        self._pre_training_actions  = []    # these lists are private to prevent callbacks from
        self._pre_epoch_actions     = []     # messing up with themselves or other callbacks
        self._pre_batch_actions     = []
        self._post_batch_actions    = []
        self._post_epoch_actions    = []
        self._post_training_actions = []

    def run(self, model: Module, training_set: DataLoader, training_callback: Optional[TrainingCallback]=None,
            validation_set: Optional[DataLoader] = None) -> None:
        """
        Runs a complete training on model. If validation_set is set to None evaluate_loss()
        will not be called.
        """
        if training_callback:
            self.add_batch_training_callback(training_callback)

        self.model = model
        self.data_loader_tr = training_set
        self.data_loader_va = validation_set
        self.model.to(self.device)  # ensure model and data are on the same device

        call_all(self._pre_training_actions)
        
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            self.current_epoch = epoch
            call_all(self._pre_epoch_actions)
            self._train_epoch()
            call_all(self._post_epoch_actions)

        call_all(self._post_training_actions)

    def attach_callback(self, callback: MTCallback) -> ModelTrainer:
        callback.trainer = self

        if callback.event is Event.ON_TRAINING_BEGIN:
            self._pre_training_actions.append(callback)
        elif callback.event is Event.ON_TRAINING_END:
            self._post_training_actions.append(callback)
        elif callback.event is Event.ON_EPOCH_BEGIN:
            self._pre_epoch_actions.append(callback)
        elif callback.event is Event.ON_EPOCH_END:
            self._post_epoch_actions.append(callback)
        elif callback.event is Event.ON_BATCH_BEGIN:
            self._pre_batch_actions.append(callback)
        elif callback.event is Event.ON_BATCH_END:
            self._post_batch_actions.append(callback)
        else:
            raise ValueError()

        callback.on_attach()
        return self

    def add_batch_training_callback(self, callback: TrainingCallback) -> ModelTrainer:
        """ 
        Adds to ModelTrainer a TrainingCallback which only input argument
        is the batch as given by the training data loader provided to ModelTrainer.
        The callback can access all ModelTrainer fields through self.trainer.
        For example a common use case for a supervised labeling task could be:

        class ExampleSupervisedBatchTrainer(TrainingCallback):
            def __call__(self, model, batch, loss_fn) -> torch.Tensor:
                data, labels = batch
                output = model(data)
                return loss_fn(output, labels)

        The callback must return a torch.Tensor representing the loss for the given batch.
        ModelTrainer will then take care of the backpropagation an weights update.
        """
        callback.trainer = self
        self._batch_training = callback
        callback.on_attach()
        return self

    # ----- private functions -----

    def _train_epoch(self) -> None:
        for batch_index, data in enumerate(self.data_loader_tr, 0):
            self.current_batch = batch_index
            if torch.is_tensor(data):
                data = data.to(self.device)
            else:   # if data is not a torch.Tensor assume the DataLoader has returned a list or tuple (data, label)
                data = [d.to(self.device) for d in data]
            
            call_all(self._pre_batch_actions)

            self.optimizer.zero_grad()  # reset parameters gradient
            batch_loss = self._batch_training(self.model, data, self.loss_fn)
            batch_loss.backward()        
            self.optimizer.step()

            self.last_batch_loss = batch_loss.item()
            call_all(self._post_batch_actions)
