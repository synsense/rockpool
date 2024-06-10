import torch
from torch import nn
import inspect
from typing import Optional, List
from torch import Tensor

from .base_loss import _Loss

__all__ = ["PeakLoss", "BinaryPeakLoss"]


class PeakLoss(_Loss):
    """
    This implements the peak loss for multiple readout neurons. The neuron with the highest output at any time point
    is interpreted as the predicting neuron and the time point of the highest output as the event time.
    The loss has two components, the loss of the target neuron and the loss of all other neurons.
    The total loss is calculated as the weighted sum of these two components.
    The loss of the target neuron is calculated and averaged over a time window starting at event time with a
    window size of max_interval. The target loss is given by the MSE loss of a specified target output
    (target_output) and the predicted output of the target neurons.
    The loss of the non-target neurons is calculated over the entire sample duration and given by the MSE loss of
    a specified non-target output (nontarget_output, default=0) and the predicted output of the non-target neurons.
    The non-target loss is also averaged over all non-target neurons.

    L = target_loss + weight_nontarget * 1/(n_classes-1) * \sum_i nontarget_loss_i

    """

    def __init__(
        self,
        max_interval: int,
        weight_nontarget: float,
        target_output: float,
        nontarget_output: Optional[float] = 0.0,
    ):
        """

        :param max_interval: window size of loss function
        :param weight_nontarget: scaling factor of loss calculated from non-target neurons
        :param target_output: target output signal of target neuron
        """

        super().__init__()
        self.max_interval = max_interval
        self.weight_nontarget = weight_nontarget
        self.target_output = target_output
        self.nontarget_output = nontarget_output
        self.mse = nn.MSELoss()
        self.loss = 0
        self.loss_target = 0
        self.loss_nontarget = 0

    def log_params(self, parser) -> None:
        params = [
            i
            for i in inspect.getmembers(self)
            if not i[0].startswith("_") and not inspect.ismethod(i[1])
        ]
        for key, value in params:
            parser.add_argument("--" + key, type=type(value), default=value)

    def calculate_loss_target(self, prediction: Tensor, target: Tensor) -> None:
        nr_samples = prediction.size()[0]
        nr_time_steps = prediction.size()[1]
        batch_indices = torch.arange(nr_samples).long()

        # pick the output of the target neurons
        prediction_target = prediction[batch_indices, :, target]
        # get the index of maximum value of the output
        max_indices_start = prediction_target.max(1)[1]
        max_indices = torch.clone(max_indices_start)

        prediction_max_target = (
            prediction_target[batch_indices, max_indices] / self.max_interval
        )
        for i in range(1, self.max_interval):
            max_indices = torch.clip(max_indices + 1, max=nr_time_steps)
            # when we reach the end of the array we jump back to the first max_index
            max_indices[max_indices == nr_time_steps] = max_indices_start[
                max_indices == nr_time_steps
            ]
            prediction_max_target += (
                prediction_target[batch_indices, max_indices] / self.max_interval
            )

        # target readout neuron should be active
        self.loss_target = self.mse(
            prediction_max_target,
            torch.ones_like(prediction_max_target) * self.target_output,
        )

    def calculate_loss_nontarget(self, prediction: Tensor, target: Tensor) -> None:
        n_classes = prediction.size()[2]
        nr_samples = prediction.size()[0]
        batch_indices = torch.arange(nr_samples).long()

        loss = 0
        # other readout neuron should be silent in the entire sample
        for i in range(1, n_classes):
            prediction_nontarget = prediction[
                batch_indices, :, (target + i) % n_classes
            ]
            loss += self.mse(
                prediction_nontarget, torch.zeros_like(prediction_nontarget)
            )

        self.loss_nontarget = loss / (n_classes - 1)

    def forward(self, prediction: Tensor, target: Tensor) -> float:
        self.calculate_loss_target(prediction, target)
        self.calculate_loss_nontarget(prediction, target)

        # add losses
        self.loss = self.loss_target + self.weight_nontarget * self.loss_nontarget
        return self.loss


class BinaryPeakLoss(_Loss):
    """
    This implements the peak loss for a single readout neurons.
    The time point of the highest output as the event time.
    The loss has two components, the loss of the positive samples and the loss of the negative samples
    The total loss is calculated as the weighted sum of these two components.
    The loss of the positive samples is calculated and averaged over a time window starting at event time with a
    window size of max_interval. The positive loss is given by the MSE loss of a specified target output
    (target_output) and the predicted output of the readout neurons.
    The loss of the negative samples is calculated over the entire sample duration and given by the MSE loss of
    a specified non-target output (nontarget_output, default=0) and the predicted output of the readout neuron.
    The negative loss is also averaged over all non-target neurons.

    L = positive_loss + weight_negatives * negative_loss

    """

    def __init__(
        self,
        max_interval: int,
        weight_nontarget: float,
        target_output: float,
        nontarget_output: Optional[float] = 0.0,
    ):
        """

        :param max_interval: window size of loss function
        :param weight_nontarget: scaling factor of loss calculated from negative samples
        :param target_output: target output signal of target neuron
        """

        super().__init__()
        self.max_interval = max_interval
        self.weight_nontarget = weight_nontarget
        self.target_output = target_output
        self.nontarget_output = nontarget_output
        self.mse = nn.MSELoss()
        self.loss = 0
        self.loss_target = 0
        self.loss_nontarget = 0

    def log_params(self, parser) -> None:
        params = [
            i
            for i in inspect.getmembers(self)
            if not i[0].startswith("_") and not inspect.ismethod(i[1])
        ]
        for key, value in params:
            parser.add_argument("--" + key, type=type(value), default=value)

    def calculate_loss_positives(self, prediction: Tensor, target: Tensor) -> None:
        nr_time_steps = prediction.size()[1]

        if sum(target == 1) > 0:
            prediction_positives = prediction[target == 1, :, 0]
            batch_indices_positives = torch.arange(
                prediction_positives.size()[0]
            ).long()
            max_indices_start = prediction_positives.max(1)[1]
            max_indices = torch.clone(max_indices_start)

            prediction_max_positives = (
                prediction_positives[batch_indices_positives, max_indices]
                / self.max_interval
            )
            for i in range(1, self.max_interval):
                max_indices = torch.clip(max_indices + 1, max=nr_time_steps)
                # when we reach the end of the array we jump back to the first max_index
                max_indices[max_indices == nr_time_steps] = max_indices_start[
                    max_indices == nr_time_steps
                ]
                prediction_max_positives += (
                    prediction_positives[batch_indices_positives, max_indices]
                    / self.max_interval
                )

            # target readout neuron should be active
            self.loss_positives = self.mse(
                prediction_max_positives,
                torch.ones_like(prediction_max_positives) * self.target_output,
            )
        else:
            self.loss_positives = 0

    def calculate_loss_negatives(self, prediction: Tensor, target: Tensor) -> None:
        # for negative samples the neuron should be silent
        if sum(target == 0) > 0:
            prediction_negatives = prediction[target == 0, :, 0]
            self.loss_negatives = self.mse(
                prediction_negatives, torch.zeros_like(prediction_negatives)
            )
        else:
            self.loss_negatives = 0

    def forward(self, prediction: Tensor, target: Tensor) -> float:
        self.calculate_loss_positives(prediction, target)
        self.calculate_loss_negatives(prediction, target)

        # add losses
        self.loss = self.loss_positives + self.weight_nontarget * self.loss_negatives
        return self.loss
