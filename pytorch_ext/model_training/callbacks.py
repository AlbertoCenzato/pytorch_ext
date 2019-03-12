import os
import time
import datetime
from typing import Tuple

import torch

from .model_trainer import MTCallback, TrainingCallback, Event
from ..visdom_board import get_visdom_manager


class SupervisedTraining(TrainingCallback):
    """
        Trains a model that receives as input an (input data, label) tuple.
        Useful in most supervised training scenarios.
    """

    def __call__(self, data_label_tuple: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        data, label = data_label_tuple
        output = self.trainer.model(data)
        return self.trainer.loss_fn(output, label)


class RNNSequenceTrainer(TrainingCallback):
    """
    Trains a recurrent model that receives as input the whole sequence
    """

    def __call__(self, data_batch: torch.Tensor) -> torch.Tensor:
        output, _ = self.trainer.model(data_batch)
        return self.trainer.loss_fn(output[:-1], data_batch[1:])


class RNNStepByStepTrainer(TrainingCallback):
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


class RNNLongTermPredictionEvaluator(MTCallback):

    BUFFERED_FRAMES = 4

    def __init__(self, early_stopping_threshold: float=0.008):
        super(RNNLongTermPredictionEvaluator, self).__init__()
        self.event = Event.ON_EPOCH_END
        self.requires_grad = []
        self.early_stop_thr = early_stopping_threshold
        self.valid_loss_plot = None

    def on_attach(self):
        vm = self.trainer.vm
        self.valid_loss_plot = vm.get_line_plot(
                                    env='Training',
                                    title='Validation loss',
                                    xaxis='epochs',
                                    yaxis='loss'
                                )

    def __call__(self) -> None:
        if self.trainer.validation_data_loader:
            loss = self.evaluate_loss(self.trainer.current_epoch)
            if loss < self.early_stop_thr:
                self.trainer.epochs = self.trainer.current_epoch

    def evaluate_loss(self, epoch: int) -> float:
        model = self.trainer.model
        if hasattr(model, 'set_mode'):
            old_mode = model.set_mode('step-by-step')

        self._freeze_model(model)

        total_loss = 0.0
        for data in iter(self.trainer.validation_data_loader):
            data = data.to(self.trainer.device)
            sequence_len = data.size(1)
            sequence_loss = 0.0
            state = model.init_hidden(data.size(0))
            for t in range(sequence_len - 1):
                if t < RNNLongTermPredictionEvaluator.BUFFERED_FRAMES:
                    output, state = model(data[:, t, :], state)
                else:
                    output, state = model(output, state)
                sequence_loss += self.trainer.loss_fn(output, data[:, t+1, :]).item()
            total_loss += sequence_loss / (sequence_len - 1)
            
        loss = total_loss/len(self.trainer.validation_data_loader)
        self.valid_loss_plot.append([epoch+1], [loss])

        if hasattr(model, 'set_mode'):
            model.set_mode('sequence')

        self._unfreeze_model(model)
        if hasattr(model, 'set_mode'):
            model.set_mode(old_mode)
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


class TrainingTimeEstimation(MTCallback):
    """
    This callback estimates the remaining training time and shows it on a visdom_board console
    """

    def __init__(self):
        super(MTCallback, self).__init__()
        self.event = Event.ON_EPOCH_BEGIN
        self.epoch_start_time = None
        self.cumulative_epochs_times = 0.0
        self.console = None

    def on_attach(self):
        vm = self.trainer.vm
        self.console = vm.get_output_console(env='Training')

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


class BatchStatistics(MTCallback):
    """
    Post-batch callback that plots the training loss
    """

    def __init__(self, period: int):
        """
        :param period: plotting period expressed in number of batches
        """
        self.event = Event.ON_BATCH_END
        self.logging_period = period
        self.running_loss = 0.0
        self.dataset_size = 0

        self.console   = None
        self.loss_plot = None

    def on_attach(self) -> None:
        vm = self.trainer.vm
        with vm.environment('Training'):
            self.console   = vm.get_output_console()
            self.loss_plot = vm.get_line_plot(
                                title='Training loss',
                                xaxis='epochs',
                                yaxis='loss'
                            )

    def __call__(self) -> None:
        batch = self.trainer.current_batch
        if batch == 0:
            self.running_loss = 0.0    # reset running loss at epoch begin

        self.running_loss += self.trainer.last_batch_loss
        if (batch % self.logging_period) == self.logging_period-1:  # print every time one-tenth of the dataset has been processed
            epoch = self.trainer.current_epoch
            dataset_size = len(self.trainer.data_loader_tr)
            mean_loss = self.running_loss / self.logging_period  # print mean loss over the processed batches
            self.console.print('[epoch: {:.0f}, batch: {:.0f}] - loss: {:.3f}'
                               .format(epoch + 1, batch + 1, mean_loss))
            self.loss_plot.append([epoch + batch / dataset_size], [mean_loss])
            self.running_loss = 0.0


class Checkpoint(MTCallback):
    """
    Callback for checkpointing the model during training.
    """

    def __init__(self, checkpoint_dir: str, period: int):
        """
        :param checkpoint_dir: path to the folder where checkpoints should be saved
        :param period: checkpointing period expressed in epochs (i.e. period=5 will create a checkpoint every 5 epochs)
        """
        super(Checkpoint, self).__init__()
        self.event = Event.ON_EPOCH_END
        self.checkpoint_dir = checkpoint_dir
        self.period = period
        self.checkpoint_counter = 0
        if not (os.path.exists(self.checkpoint_dir) and os.path.isdir(self.checkpoint_dir)):
            os.mkdir(self.checkpoint_dir)

    def __call__(self):
        if self.trainer.current_epoch % self.period == 0:
            model_path = os.path.join(self.checkpoint_dir, 'checkpoint_{}.pt'.format(self.checkpoint_counter))
            torch.save(self.trainer.model, model_path)
            self.checkpoint_counter += 1


class ProgressiveNetInspector(Checkpoint):
    """
    This callback takes a checkpoint of the model and shows it in VisdomBoard
    """

    def __init__(self, batch_first, *args):
        super(ProgressiveNetInspector, self).__init__(*args)
        self.batch_dim = 0 if batch_first else 1
        self.vm = get_visdom_manager()
        self.net_inspector = None
        self.test_tensor   = None

    def __call__(self):
        super(ProgressiveNetInspector, self).__call__()

        if self.test_tensor is None:
            self.test_tensor = iter(self.trainer.data_loader_tr).next()
            if self.test_tensor.size(self.batch_dim) > 1:  # we want batch_size = 1
                self.test_tensor = self.test_tensor.index_select(dim=self.batch_dim,
                                                                 index=torch.zeros((1,), dtype=torch.long))

        if self.trainer.current_epoch % self.period == 0:
            # load checkpointed model
            model_path = os.path.join(self.checkpoint_dir, 'checkpoint_{}.pt'.format(self.checkpoint_counter-1))
            model = torch.load(model_path).cpu()  # NOTE: if the model was trained on GPU the model is first loaded in GPU, then moved to cpu
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            if hasattr(model, 'set_mode'):
                model.set_mode('sequence')

            if self.net_inspector:
                self.net_inspector.close()
            self.net_inspector = self.vm.get_net_inspector(model, self.test_tensor)



