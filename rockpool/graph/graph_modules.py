from rockpool.graph.graph_base import GraphModule

from dataclasses import dataclass
from typing import Any

ArrayLike = Any

import numpy as np

__all__ = ["LinearWeights", "GenericNeurons"]


@dataclass(eq=False, repr=False)
class LinearWeights(GraphModule):
    weights: ArrayLike

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

        # - Convert weights to numpy array
        self.weights = np.array(self.weights)

        # - Check size
        if self.weights.shape != (len(self.input_nodes), len(self.output_nodes)):
            raise ValueError(
                f"`weights` must match size of input and output nodes. Got {self.weights.shape}, expected {(len(self.input_nodes), len(self.output_nodes))}."
            )

        # - Attach input and output nodes back to module
        for n in self.input_nodes:
            n.add_target(self)

        for n in self.output_nodes:
            n.add_source(self)


@dataclass(eq=False, repr=False)
class GenericNeurons(GraphModule):
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

        # - Attach input and output nodes back to module
        for n in self.input_nodes:
            n.add_target(self)

        for n in self.output_nodes:
            n.add_source(self)


@dataclass(eq=False, repr=False)
class AliasConnection(GraphModule):
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

        # - Attach input and output nodes back to module
        for n in self.input_nodes:
            n.add_target(self)

        for n in self.output_nodes:
            n.add_source(self)
