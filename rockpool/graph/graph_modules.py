"""
Basic computational modules for graph definition in Rockpool

Defines :py:class:`.LinearWeights`, :py:class:`.GenericNeurons`, :py:class:`.AliasConnection` and :py:class:`.LIFNeuronRealValue`.
"""


from rockpool.graph.graph_base import GraphModule

from dataclasses import dataclass, field
from typing import Optional

from rockpool.typehints import FloatVector

import numpy as np
from rockpool.utilities.backend_management import backend_available

if backend_available("torch"):
    from torch import Tensor
else:

    class Tensor:
        pass


__all__ = [
    "LinearWeights",
    "GenericNeurons",
    "AliasConnection",
    "LIFNeuronWithSynsRealValue",
    "RateNeuronWithSynsRealValue",
]


@dataclass(eq=False, repr=False)
class LinearWeights(GraphModule):
    """
    A :py:class:`.GraphModule` that encapsulates a single set of linear weights, with no biases
    """

    weights: FloatVector
    """ FloatVector: The linear weights ``(Nin, Nout)`` encapsulated by this module """

    def __post_init__(self, *args, **kwargs):
        # - Check size
        if self.weights.shape != (len(self.input_nodes), len(self.output_nodes)):
            raise ValueError(
                f"`weights` must match size of input and output nodes. Got {self.weights.shape}, expected {(len(self.input_nodes), len(self.output_nodes))}."
            )

        super().__post_init__(*args, **kwargs)

        # - Convert weights to numpy array
        if isinstance(self.weights, Tensor):
            self.weights = np.array(self.weights.detach().cpu().numpy())
        else:
            self.weights = np.array(self.weights)


@dataclass(eq=False, repr=False)
class GenericNeurons(GraphModule):
    """
    A :py:class:`.GraphModule` than encapsulates a set of generic neurons

    This class is used as a base class for all specific neuron subclasses. It defines only input and output nodes, and does not specify any parameters for the neurons.
    """

    pass


@dataclass(eq=False, repr=False)
class AliasConnection(GraphModule):
    """
    A :py:class:`.GraphModule` that encapsulates a set of alias connections
    """

    def __post_init__(self, *args, **kwargs):
        # - Call super-class
        super().__post_init__(*args, **kwargs)

        # - Check size
        if len(self.input_nodes) != len(self.output_nodes):
            raise ValueError(
                f"For an alias connection, the number of inputs and outputs must be identical.\nGot {len(self.input_nodes)} and {len(self.output_nodes)}."
            )


@dataclass(eq=False, repr=False)
class LIFNeuronWithSynsRealValue(GenericNeurons):
    """
    A :py:class:`.GraphModule` that encapsulates a set of LIF spiking neurons with synaptic and membrane dynamics, and with real-valued parameters
    """

    tau_mem: FloatVector = field(default_factory=list)
    """ Floatvector: The membrane time constants of these neurons, in seconds ``(Nout,)`` """

    tau_syn: FloatVector = field(default_factory=list)
    """ Floatvector: The synaptic time constants of these neurons, in seconds ``(Nin,)`` """

    threshold: FloatVector = field(default_factory=list)
    """ Floatvector: The firing threshold parameters of these neurons ``(Nout,)`` """

    bias: FloatVector = field(default_factory=list)
    """ Floatvector: The bias parameters of these neurons, if present ``(Nout,)`` """

    dt: Optional[float] = None
    """ float: The time-step used for these neurons in seconds, if present """


@dataclass(eq=False, repr=False)
class RateNeuronWithSynsRealValue(GenericNeurons):
    """
    A :py:class:`.GraphModule` that encapsulates a set of rate neurons, with synapses, and with real-valued parameters
    """

    tau: FloatVector = field(default_factory=list)
    """ Floatvector: The time constants of these neurons, in seconds ``(Nout,)`` """

    bias: FloatVector = field(default_factory=list)
    """ Floatvector: The bias parameters of these neurons ``(Nout,)`` """

    dt: Optional[float] = None
    """ float: The time-step used for these neurons in seconds, if present """
