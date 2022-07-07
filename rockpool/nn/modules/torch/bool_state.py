"""
A module implementing a boolean state variable
"""

from rockpool.nn.modules.torch import TorchModule
from rockpool.parameters import Parameter, SimulationParameter, State
from rockpool.typehints import FloatVector

from typing import Tuple

import torch

__all__ = ["BooleanState"]


class BooleanState(TorchModule):
    """
    This module implements a boolean state, which is flipped between ``True`` or ``False`` depending on the input.

    If ``input`` > a threshold (1e-10 by default), the corresponding state will be flipped to ``True`.

    If ``input`` < -threshold, the corresponding state will be flipped to ``False``.

    Otherwise the state will remain unchanged.

    On initialisation the states are set to ``False``.
    """

    def __init__(
        self,
        shape: Tuple[int],
        threshold: FloatVector = None,
        *args,
        **kwargs,
    ):
        """
        Initialise a boolean state module

        Args:
            shape (Tuple): The shape of this module
            threshold (FloatVector): The threshold used to flip states. Default: 1e-10.
        """
        # - Initialise superclass
        super().__init__(
            shape=shape, spiking_input=False, spiking_output=False, *args, **kwargs
        )

        if self.size_in != self.size_out:
            raise ValueError("`size_in` must be equal to `size_out` for BooleanState.")

        if threshold is not None:
            if torch.any(threshold < 0):
                raise ValueError("threshold must be positive")

        # - Initialise threshold
        self.threshold = SimulationParameter(
            threshold,
            shape=[(), (self.size_out,)],
            init_func=lambda s: 1e-10 * torch.ones(s),
        )

        # - Initialise state
        self.bool_state = State(
            shape=(self.size_out,), init_func=lambda s: torch.zeros(s, dtype=torch.bool)
        )

        # - Initialise recording dictionary
        self._record_dict = {}

    def forward(self, input_data: torch.Tensor):
        # - Perform auto-batching over data
        data, (bool_state,) = self._auto_batch(input_data, (self.bool_state,))
        (num_batches, num_timesteps, _) = data.shape

        # - Convert input to a sign tensor, taking into account threshold
        bdata = (torch.abs(data) > self.threshold) * torch.sign(data)

        # - Intialise state trace
        this_state = torch.empty(
            num_batches,
            num_timesteps + 1,
            self.size_out,
            dtype=bool,
            device=data.device,
        )
        this_state[:, 0, :] = bool_state

        for t in range(num_timesteps):
            this_state[:, t + 1, :] = (
                torch.clip(bdata[:, t, :] + this_state[:, t, :], 0, torch.inf) > 0
            )

        # - Record state trace
        self._record_dict["bool_state"] = this_state[:, 1:, :]

        # - Record state
        self.bool_state = this_state[0, -1, :]

        # - Return output trace
        return self._record_dict["bool_state"]
