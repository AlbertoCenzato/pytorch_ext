import os
import time
import datetime
from typing import Tuple

import torch

from ..visdom_board import get_visdom_manager
from .model_trainer import TrainingCallback, BatchTrainingCallback


class SupervisedTraining(BatchTrainingCallback):
    """
        Trains a model that receives as input an (input data, label) tuple.
        Useful in most supervised training scenarios.
    """

    def __call__(self, data_label_tuple: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        data  = data_label_tuple[0]
        label = data_label_tuple[1]
        output = self.trainer.model(data)
        return self.trainer.loss_fn(output, label)


class RNNSequenceTrainer(BatchTrainingCallback):
    """
        Trains a recurrent model that receives as input the whole sequence
    """

    def __call__(self, data_batch: torch.Tensor) -> torch.Tensor:
        output, _ = self.trainer.model(data_batch)
        return self.trainer.loss_fn(output[:-1], data_batch[1:])


class RNNStepByStepTrainer(BatchTrainingCallback):
    """
        Trains a recurrent model that receives as input a single time frame
    """

    def __call__(self, data_batch: torch.Tensor) -> torch.Tensor:
        # checking data.size(0) at each call instead of storing self.batch_size 
        # because if (training data len % self.batch_size != 0) then the last
        # batch does not have self.batch_size elements
        state = self.trainer.model.init_hidden(data_batch.size(0))
        sequence_loss = torch.zeros(1).to(self.trainer.device)
        for t in range(data_batch.size(1)-1):
            output, state = self.trainer.model(data_batch[:, t, :], state)
            ground_truth = data_batch[:, t+1, :]
            sequence_loss += self.trainer.loss_fn(output, ground_truth)
        
        return sequence_loss



class RNNLongTermPredictionEvaluator(TrainingCallback):

    BUFFERED_FRAMES = 4

    def __init__(self, early_stopping_threshold: float=0.008):
        super(RNNLongTermPredictionEvaluator, self).__init__()
        self.requires_grad = []
        self.early_stop_thr = early_stopping_threshold
        vm = get_visdom_manager()
        self.valid_loss_plot = vm.get_line_plot(env='Training',
                                                title='Validation loss',
                                                xaxis='epochs',
                                                yaxis='loss')

    def __call__(self) -> None:
        if self.trainer.validation_data_loader:
            loss = self.evaluate_loss(self.trainer.current_epoch)
            if loss < self.early_stop_thr:
                self.trainer.epochs = self.trainer.current_epoch

    def evaluate_loss(self, epoch: int) -> float:
        if hasattr(self.trainer.model, 'set_mode'):
            old_mode = self.trainer.model.set_mode('step-by-step')

        self._freeze_model(self.trainer.model)

        total_loss = 0.0
        for data in iter(self.trainer.validation_data_loader):
            data = data.to(self.trainer.device)
            sequence_len = data.size(1)
            sequence_loss = 0.0
            state = self.trainer.model.init_hidden(data.size(0))
            for t in range(sequence_len - 1):
                if t < RNNLongTermPredictionEvaluator.BUFFERED_FRAMES:
                    output, state = self.trainer.model(data[:, t, :], state)
                else:
                    output, state = self.trainer.model(output, state)
                sequence_loss += self.trainer.loss_fn(output, data[:, t+1, :]).item()
            total_loss += sequence_loss / (sequence_len - 1)
            
        loss = total_loss/len(self.trainer.validation_data_loader)
        self.valid_loss_plot.append([epoch+1], [loss])

        if hasattr(self.trainer.model, 'set_mode'):
            self.trainer.model.set_mode('sequence')

        self._unfreeze_model(self.trainer.model)
        if hasattr(self.trainer.model, 'set_mode'):
            self.trainer.model.set_mode(old_mode)
        return loss

    def _freeze_model(self, model: torch.nn.Module) -> None:
        model.eval()
        self.requires_grad = []
        for param in model.parameters():  # WARNING: looping in this way assumes that model parameters are always yielded in the same order
            self.requires_grad.append(param.requires_grad)
            param.requires_grad = False

    def _unfreeze_model(self, model: torch.nn.Module) -> None:
        model.train()
        for i, param in enumerate(model.parameters()):  # WARNING: looping in this way assumes that model parameters are always yielded in the same order
            param.requires_grad = self.requires_grad[i]


class TrainingTimeEstimation(TrainingCallback):
    """
    This callback estimates the remaining training time and shows it on a visdom_board console
    """

    def __init__(self):
        super(TrainingCallback, self).__init__()
        vm = get_visdom_manager()
        self.console = vm.get_output_console(env='Training')
        self.epoch_start_time = None
        self.cumulative_epochs_times = 0.0

    def __call__(self) -> None:
        if not self.epoch_start_time:
            self.epoch_start_time = time.time()
        else:
            end = time.time()
            self.cumulative_epochs_times += end - self.epoch_start_time
            estimated_time_per_epoch = self.cumulative_epochs_times / self.trainer.current_epoch
            remaining_epochs = self.trainer.epochs - self.trainer.current_epoch

            self.console.clear_console()
            eta = estimated_time_per_epoch * remaining_epochs
            time_delta = datetime.timedelta(seconds=int(eta))
            self.console.print('ETA: {}'.format(time_delta))

            self.epoch_start_time = end


class Checkpoint(TrainingCallback):
    """
        Callback for checkpointing the model during training
    """

    def __init__(self, checkpoint_dir: str, period: int):
        """
        :param checkpoint_dir: path to the folder where checkpoints should be saved
        :param period: checkpointing period expressed in epochs (i.e. period=5 will create a checkpoint every 5 epochs)
        """
        super(Checkpoint, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.period = period
        self.checkpoint_counter = 0
        if not (os.path.exists(self.checkpoint_dir) and os.path.isdir(self.checkpoint_dir)):
            os.mkdir(self.checkpoint_dir)

    def __call__(self):
        if self.trainer.current_epoch % self.period == 0:
            model_path = os.path.join(self.checkpoint_dir, 'checkpoint_{}.pt'.format(self.checkpoint_counter))
            torch.save(self.trainer.model, model_path)
