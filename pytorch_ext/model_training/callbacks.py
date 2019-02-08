import random
import time
import datetime
from collections import deque
from typing import Tuple, Union

import torch

from visdom_board import get_visdom_manager
from .model_trainer import TrainingCallback, BatchTrainingCallback


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


class RNNCurriculumLearningTrainer(BatchTrainingCallback):
    """
    """
    class MovingAverage:

        def __init__(self, max_len: int):
            self.max_len = max_len
            self.past_elements = deque()
            self.moving_sum = 0.0

        def append(self, elem: Union[int, float]) -> None:
            if len(self.past_losses) >= self.max_len:
                self.moving_sum -= self.past_elements.popleft()
            self.moving_sum += elem
            self.past_losses.append(elem)

        def get(self) -> float:
            return self.moving_sum / len(self.past_losses)

        def full(self) -> bool:
            return len(self) == self.max_len

        def __len__(self) -> int:
            return len(self.past_elements)

    def __init__(self, threshold: float, max_predicted_frames: int,
                 max_under_threshold_batches: int=50):
        super(RNNCurriculumLearningTrainer, self).__init__()
        self.threshold = threshold
        self.max_predicted_frames = max_predicted_frames
        self.n_of_predicted_frames = 1

        self.under_threshold_batches = 0
        self.max_under_threshold_batches = max_under_threshold_batches

        self.moving_average = RNNCurriculumLearningTrainer.MovingAverage(self.max_under_threshold_batches)

    def __call__(self, data_batch: torch.Tensor) -> torch.Tensor:
        # using data.size(0) instead of self.batch_size because 
        # if training_data_len % self.batch_size != 0 then the last
        # batch returned by enumerate() does not have self.batch_size elements
        model = self.trainer.model
        state = model.init_hidden(data_batch.size(0))
        sequence_len = data_batch.size(1)
                
        gen_start, gen_end = self._naive_curriculum_learning(buffering_frames=4, sequence_len=sequence_len)

        sequence_loss = torch.zeros(1).to(self.trainer.device)
        for t in range(sequence_len - 1):
            if gen_start <= t <= gen_end:
                output, state = model(output, state)
            else:
                output, state = model(data_batch[:, t, :], state)
            ground_truth = data_batch[:, t+1, :]
            sequence_loss += self.trainer.loss_fn(output, ground_truth)

        self._raise_difficulty(sequence_loss.item())

        return sequence_loss

    def _raise_difficulty(self, loss: float) -> None:
        if self.moving_average.full():
            average = self.moving_average.get()
            if average - self.threshold <= loss <= average + self.threshold:
                self.n_of_predicted_frames = min(self.n_of_predicted_frames + 1, self.max_predicted_frames)
                print('Loss has not changed for {} batches: raising difficulty'
                      .format(self.threshold, self.max_under_threshold_batches))

        self.moving_average.append(loss)

    def _naive_curriculum_learning(self, buffering_frames: int, sequence_len: int) -> Tuple[int, int]:
        """ 
            Returns begin and end indexes for the n-frames ahead predictions
            following the naive curriculum learning strategy used in 
            Zaremba and Sustskever, 'Learning to execute', 2014.

            Output: (begin_index, end_index).
            buffering_frames <= begin_index <= end_index < sequence_len
        """
        begin_index = random.randint(buffering_frames, sequence_len - self.n_of_predicted_frames - 1)
        end_index   = begin_index + self.n_of_predicted_frames
        return begin_index, end_index

    def _combined_curriculum_learning(self, buffering_frames: int, sequence_len: int) -> Tuple[int, int]:
        """ 
            Returns begin and end indexes for the n-frames ahead predictions
            following the combined curriculum learning strategy used in 
            Zaremba and Sustskever, 'Learning to execute', 2014.

            Output: (begin_index, end_index).
            buffering_frames <= begin_index <= end_index < sequence_len
        """

        strategy = 'naive' if random.randint(0,99) < 80 else 'mixed'
        if strategy == 'naive':
            begin_index, end_index = self._naive_curriculum_learning(buffering_frames, sequence_len)
        else:
            begin_index = random.randint(buffering_frames, sequence_len-1)
            end_index   = random.randint(begin_index, sequence_len - 1)

        return begin_index, end_index


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