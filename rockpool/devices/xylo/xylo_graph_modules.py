from rockpool.graph import (
    GenericNeurons,
    GraphModule,
    LIFNeuronRealValue,
    replace_module,
)

import numpy as np

from typing import List

from dataclasses import dataclass, field

__all__ = ["XyloNeurons", "XyloHiddenNeurons", "XyloOutputNeurons"]


@dataclass(eq=False, repr=False)
class XyloNeurons(GenericNeurons):
    hw_ids: List[int] = field(default_factory=list)
    threshold: List[int] = field(default_factory=list)
    dash_mem: List[int] = field(default_factory=list)
    dash_syn: List[int] = field(default_factory=list)
    dt: float = None

    @classmethod
    def _convert_from(cls, mod: GraphModule) -> GraphModule:
        if isinstance(mod, cls):
            # - No need to do anything
            return mod

        elif isinstance(mod, LIFNeuronRealValue):
            # - Get values for TCs
            if mod.dt is None:
                raise ValueError(
                    f"Graph module of type {type(mod).__name__} has no `dt` set, so cannot convert time constants when converting to {cls.__name__}."
                )

            dash_mem = (
                np.round(np.log2(np.array(mod.tau_mem) / mod.dt)).astype(int).tolist()
            )
            dash_syn = (
                np.round(np.log2(np.array(mod.tau_syn) / mod.dt)).astype(int).tolist()
            )
            thresholds = np.round(np.array(mod.threshold)).astype(int).tolist()

            neurons = cls._factory(
                len(mod.input_nodes),
                len(mod.output_nodes),
                mod.name,
                [],
                thresholds,
                dash_mem,
                dash_syn,
                mod.dt,
            )

            # - Replace the module and return
            replace_module(mod, neurons)
            return neurons

        elif isinstance(mod, GenericNeurons):
            # - Make a new module
            neurons = cls._factory(
                len(mod.input_nodes),
                len(mod.output_nodes),
                mod.name,
            )

            # - Replace the module
            replace_module(mod, neurons)

            # - Try to set attributes of the new module
            for attr in neurons.__dataclass_fields__.keys():
                if hasattr(mod, attr):
                    setattr(neurons, attr, getattr(mod, attr))

            return neurons

        else:
            raise ValueError(
                f"Graph module of type {type(mod).__name__} cannot be converted to a {cls.__name__}"
            )


@dataclass(eq=False, repr=False)
class XyloHiddenNeurons(XyloNeurons):
    def __post_init__(self, *args, **kwargs):
        if len(self.input_nodes) != len(self.output_nodes):
            if len(self.input_nodes) != 2 * len(self.output_nodes):
                raise ValueError(
                    "Number of input nodes must be 1* or 2* number of output nodes"
                )

        super().__post_init__(self, *args, **kwargs)


@dataclass(eq=False, repr=False)
class XyloOutputNeurons(XyloNeurons):
    def __post_init__(self, *args, **kwargs):
        if len(self.input_nodes) != len(self.output_nodes):
            raise ValueError(
                "Number of input nodes must be equal to number of output nodes"
            )

        super().__post_init__(self, *args, **kwargs)
