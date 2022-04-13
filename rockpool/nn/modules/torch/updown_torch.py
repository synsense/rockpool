"""
Feedforward layer that converts each analogue input channel to one spiking up and one down channel
"""
from typing import Optional, Union, Tuple, Any
import numpy as np
from rockpool.nn.modules.torch.torch_module import TorchModule
import torch
import rockpool.parameters as rp
from rockpool.typehints import FloatVector, P_float, P_tensor

__all__ = ["UpDownTorch"]


class StepPWL(torch.autograd.Function):
    """
    Heaviside step function with piece-wise linear derivative to use as spike-generation surrogate

    :param torch.Tensor data: Input data
    :param float thr: Threshold value

    :return torch.Tensor: output value and gradient function
    """

    @staticmethod
    def forward(ctx, data):
        ctx.save_for_backward(data)
        return torch.clamp(torch.floor(data), 0)

    @staticmethod
    def backward(ctx, grad_output):
        (data,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[data < 0.5] = 0
        # - Since there are two inputs, we need to give two outputs to backpropagate.
        return grad_input


## - UpDownTorch - Class: Define a spiking feedforward layer to convert analogue inputs to up and down channels
class UpDownTorch(TorchModule):
    """
    Feedforward layer that converts each analogue input channel to one spiking up and one spiking down channel.

    This module orients itself on the ADC implementation in 'A Neuromorphic Event-Based Neural Recording System for Smart Brain-Machine-Interfaces', Corradi et al. 2015. While following the same idea, the dynamics and non-idealities of the module are not modeled, instead this module strives to be an nominal implementation of the idea of an up-down ADM.

    The spike generation is dependent on whether the input value surpasses/falls below the up/down threshold relative of the reference value. If a threshold is reached, a spike will be emitted and the threshhold value added (in case of an up spike) resp. subtracted from the refernece value.

    This module also allows for setting a refractory period, which is activated after a spike was emitted on either of the output channels, during which further spike emitting is supressed.
    """

    ## - Constructor
    def __init__(
        self,
        shape: tuple = None,
        thr_up: Optional[FloatVector] = 1e-3,
        thr_down: Optional[FloatVector] = 1e-3,
        n_ref_steps: int = 0,
        repeat_output: int = 1,
        dt: float = 1e-3,
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ):
        """
        Instantiate a spiking feedforward layer to convert analogue inputs to up and down spike channels.

        Args:
            shape (tuple): A single dimension ``(N_in,)``, which defines the number of input channels. The output is always given as ``N_out = 2 * N_in``.
            thr_up (Optional[FloatVector]): Thresholds for creating up-spikes. Default: ``0.001``
            thr_down (Optional[FloatVector]): Thresholds for creating down-spikes. Default: ``0.001``
            n_ref_steps (float): Determines the duration of the refractory period as multiple of `dt` (`t_ref=n_ref_steps*dt`). During the refractory period the module doesn't emit any spikes. Default: ``0``
            repeat_output (int): Repeat each output spike x times.
            dt (float): The time step for the forward-Euler ODE solver in seconds. Default: ``1ms``
            device: Defines the device on which the model will be processed.
            dtype: Defines the data type of the tensors saved as attributes.
            device: Defines the device on which the module will be processed.
            dtype: Defines the data type of the tensors saved as attributes.
        """

        if np.size(shape) == 1:
            shape_in = np.array(shape).item()
            shape = (shape_in, 2 * shape_in)
        else:
            raise ValueError("`shape` must be a one-element tuple `(Nin,)`.")

        # - Call super constructor
        super().__init__(
            shape=shape,
            spiking_input=False,
            spiking_output=True,
            *args,
            **kwargs,
        )

        # - Default tensor construction parameters
        self._factory_kwargs = {"device": device, "dtype": dtype}

        # - Store layer parameters
        self.repeat_output: P_float = rp.SimulationParameter(repeat_output)
        self.n_ref_steps: P_float = rp.SimulationParameter(n_ref_steps)

        if np.size(thr_up) == 1:
            thr_up = torch.ones((1, self.size_in), **self._factory_kwargs) * thr_up
        else:
            thr_up = thr_up.view(1, -1)

        self.thr_up: P_tensor = rp.Parameter(
            thr_up,
            family="thresholds",
        )
        """ (Tensor) Thresholds for creating up-spikes `(N_in,)` """

        if np.size(thr_down) == 1:
            thr_down = torch.ones((1, self.size_in), **self._factory_kwargs) * thr_down
        else:
            thr_down = thr_up.view(1, -1)

        self.thr_down: P_tensor = rp.Parameter(
            thr_down,
            family="thresholds",
        )
        """ (Tensor) Thresholds for creating down-spikes `(N_in,)` """

    def evolve(
        self,
        input_data: torch.Tensor,
        record: bool = False,
    ) -> Tuple[Any, Any, Any]:
        # - Evolve with superclass evolution
        output_data, states, _ = super().evolve(input_data, record)

        # - Build state record
        record_dict = (
            {
                "analog_value": self._analog_value_rec,
            }
            if record
            else {}
        )

        return output_data, states, record_dict

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward method for processing data through this layer
        Convert each analog input channel to an up and down spike channel.

        ----------
        data: Tensor
            Data takes the shape of `(batch, time_steps, n_channels)`

        Returns
        -------
        out: Tensor
            Output of spikes with the shape `(batch, time_steps, 2*n_channels)`, where the `2*n`-th output channel the up channel and the `(2*n + 1)`-th output channel the down channel of the `n`-th input channel are.

        """
        n_batches, time_steps, n_channels = data.shape

        if n_channels != self.size_in:
            raise ValueError(
                "Input has wrong input channel dimension. It is {}, must be {}".format(
                    n_channels, self.size_in
                )
            )

        # - Extend thresholds by batches
        thr_up = torch.ones(n_batches, 1) @ self.thr_up
        thr_down = torch.ones(n_batches, 1) @ self.thr_down

        """ Counter, for how many steps of dt is the module still in refractory period.
        Has to be counted for each batch and channel individually. """
        remaining_ref_steps = torch.zeros(n_batches, n_channels)

        # - Reference value from where we observe whether the signal surpasses any thresholds
        analog_value = data[:, 0, :].detach()

        step_pwl = StepPWL.apply

        # - Set up state record and output
        self._analog_value_rec = torch.zeros(
            n_batches, time_steps, n_channels, **self._factory_kwargs
        )
        out_spikes = torch.zeros(
            n_batches, time_steps, self.size_out, **self._factory_kwargs
        )

        # - Loop over time
        for t in range(time_steps):
            # - Record the state
            self._analog_value_rec[:, t, :] = analog_value.detach()

            # - Get the difference between the last analog value saved
            diff_values = data[:, t, :] - analog_value

            # - Calculate the spike outputs
            up_channels = step_pwl(diff_values / thr_up)
            # - Enter the negative thr_down so that it checks for changes going below this threshold.
            down_channels = step_pwl(diff_values / (-thr_down))

            if self.n_ref_steps > 0:
                # - Remove the spikes of all channels that are still in the refractory period
                up_channels[remaining_ref_steps > 0] = 0
                down_channels[remaining_ref_steps > 0] = 0

                # - Limit the amount of emitted spikes to 1, since the refractory period supresses all spikes after the first one
                up_channels[up_channels >= 1] = 1
                down_channels[down_channels >= 1] = 1

                # - Reset the refractory counter back to the full time when either an up or a down spike was emitted
                remaining_ref_steps[
                    (up_channels + down_channels) > 0
                ] = self.n_ref_steps

                # - Set the reference value to the last input for all channels which are in refractory period
                analog_value[remaining_ref_steps > 0] = (data[:, t, :])[
                    remaining_ref_steps > 0
                ]

                # - Count down the refractory counters
                remaining_ref_steps -= 1
            else:
                # - Add (resp. subtract) the thresholds for every emitted spike
                analog_value = analog_value + up_channels * thr_up
                analog_value = analog_value - down_channels * thr_down

            # - Interleave up and down channels so we have 2*k as up and 2*k + 1 as down channel of the k-th input channel
            # - Multiply by repeat_output to get the desired multiple of spikes
            out_spikes[:, t, :] = self.repeat_output * torch.stack(
                (up_channels, down_channels),
                dim=2,
            ).view(n_batches, 2 * n_channels)

        return out_spikes
