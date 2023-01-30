"""
A module implementing random dropout of neurons and time steps
"""

from rockpool.nn.modules.torch import TorchModule
from rockpool.parameters import SimulationParameter
from rockpool.graph import GraphModuleBase, SetList, GraphModule, GraphNode, GraphHolder


from typing import Tuple, Any

import torch

__all__ = ["UnitDropout", "TimeStepDropout"]


class UnitDropout(TorchModule):
    """
    For each gradient update units are dropped with a given probability.
    The module can be connected to continuos or spiking modules such as LinearTorch or LIFTorch.
    Dropout in Pytorch scales the output of the remaining time steps with 1/(1-self.p),
    some brief experiments with a model on the google speech command dataset showed that this is also beneficial here,
    so the output is scaled
    """

    def __init__(
        self,
        shape: Tuple[int],
        p: float = 0.5,
        training: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialise a dropout module

        Args:
            shape (Tuple): The shape of this module
            p (float): Probability that neuron is silenced. Default: 0.5 (same as in pytorch)
            training (bool): Indicates training or evaluation mode. Dropout is disabled for validation/testing.
        """
        # - Initialise superclass
        super().__init__(
            shape=shape, spiking_input=False, spiking_output=False, *args, **kwargs
        )

        if self.size_in != self.size_out:
            raise ValueError("`size_in` must be equal to `size_out` for Dropout.")

        if p >= 1.0 or p < 0.0:
            raise ValueError("dropout probability must be between zero and one")

        self.p = SimulationParameter(p)
        self.training = SimulationParameter(training)

    def forward(self, data: torch.Tensor):
        data, _ = self._auto_batch(data)
        num_batches, num_timesteps, num_neurons = data.shape

        if self.training:
            # mask input
            mask = torch.ones(
                (num_batches, num_neurons, num_timesteps), device=data.device
            )
            probs = torch.rand((num_batches, num_neurons))
            mask[probs < self.p] = 0
            mask = torch.swapaxes(mask, 1, 2)
            return data * mask * 1 / (1 - self.p)
        else:
            return data

    def as_graph(self) -> GraphModuleBase:
        n = SetList([GraphNode() for _ in range(self.size_in)])
        return GraphHolder(n, n, f"Dropout_'{self.name}'_{id(self)}", self)


class TimeStepDropout(TorchModule):
    """
    For each gradient update time steps are dropped with a given probability.
    The module can be connected to continuos or spiking modules such as LinearTorch or LIFTorch.
    Dropout in Pytorch scales the output of the remaining time steps with 1/(1-self.p),
    some brief experiments with a model on the google speech command dataset showed that this is noy beneficial here,
    so the output is not scaled
    """

    def __init__(
        self,
        shape: Tuple[int],
        p: float = 0.5,
        training: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialise a dropout module

        Args:
            shape (Tuple): The shape of this module
            p (float): Probability that neuron is silenced. Default: 0.5 (same as in pytorch)
            training (bool): Indicates training or evaluation mode. Dropout is disabled for validation/testing.
        """
        # - Initialise superclass
        super().__init__(
            shape=shape, spiking_input=False, spiking_output=False, *args, **kwargs
        )

        if self.size_in != self.size_out:
            raise ValueError("`size_in` must be equal to `size_out` for Dropout.")

        if p > 1.0 or p < 0.0:
            raise ValueError("dropout probability must be between zero and one")

        self.p = SimulationParameter(p)
        self.training = SimulationParameter(training)

    def forward(self, data: torch.Tensor):
        data, _ = self._auto_batch(data)
        num_batches, num_timesteps, num_neurons = data.shape

        if self.training:
            # mask input
            mask = torch.ones(
                (num_batches, num_timesteps, num_neurons), device=data.device
            )
            probs = torch.rand((num_batches, num_timesteps, num_neurons))
            mask[probs < self.p] = 0
            return data * mask
        else:
            return data

    def as_graph(self) -> GraphModuleBase:
        n = SetList([GraphNode() for _ in range(self.size_in)])
        return GraphHolder(n, n, f"Dropout_'{self.name}'_{id(self)}", self)
