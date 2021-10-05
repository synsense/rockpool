from rockpool.graph.graph_base import GraphModule

from dataclasses import dataclass, field
from typing import Iterable, Any, Union

ArrayLike = Iterable

import numpy as np

__all__ = ["LinearWeights", "GenericNeurons", "AliasConnection", "LIFNeuronRealValue"]


@dataclass(eq=False, repr=False)
class LinearWeights(GraphModule):
    weights: Union[np.array, Any]

    def __post_init__(self, *args, **kwargs):
        # - Check size
        if self.weights.shape != (len(self.input_nodes), len(self.output_nodes)):
            raise ValueError(
                f"`weights` must match size of input and output nodes. Got {self.weights.shape}, expected {(len(self.input_nodes), len(self.output_nodes))}."
            )

        super().__post_init__(*args, **kwargs)

        # - Convert weights to numpy array
        self.weights = np.array(self.weights)

        # - Attach input and output nodes back to module
        for n in self.input_nodes:
            n.add_sink(self)

        for n in self.output_nodes:
            n.add_source(self)


@dataclass(eq=False, repr=False)
class GenericNeurons(GraphModule):
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

        # - Attach input and output nodes back to module
        for n in self.input_nodes:
            n.add_sink(self)

        for n in self.output_nodes:
            n.add_source(self)


@dataclass(eq=False, repr=False)
class AliasConnection(GraphModule):
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

        # - Attach input and output nodes back to module
        for n in self.input_nodes:
            n.add_sink(self)

        for n in self.output_nodes:
            n.add_source(self)


@dataclass(eq=False, repr=False)
class LIFNeuronRealValue(GenericNeurons):
    tau_mem: ArrayLike[float] = field(default_factory=list)
    tau_syn: ArrayLike[float] = field(default_factory=list)
    threshold: ArrayLike[float] = field(default_factory=list)
    bias: ArrayLike[float] = field(default_factory=list)
    dt: float = None
