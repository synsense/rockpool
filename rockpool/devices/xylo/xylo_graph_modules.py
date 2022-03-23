"""
Xylo graph modules for use with tracing and mapping
"""
import warnings

from rockpool.graph import (
    GenericNeurons,
    GraphModule,
    LIFNeuronWithSynsRealValue,
    replace_module,
)

import numpy as np

from typing import List, Optional, Union
from rockpool.typehints import IntVector, FloatVector

from dataclasses import dataclass, field

__all__ = ["XyloNeurons", "XyloHiddenNeurons", "XyloOutputNeurons"]


@dataclass(eq=False, repr=False)
class XyloNeurons(GenericNeurons):
    """
    Base class for all Xylo graph module classes
    """

    hw_ids: Union[IntVector, FloatVector] = field(default_factory=list)
    """ IntVector: The HW neuron IDs allocated to this graph module ``(N,)``. Empty means than no HW IDs have been allocated."""

    threshold: Union[IntVector, FloatVector] = field(default_factory=list)
    """ IntVector: The threshold parameters for each neuron ``(N,)`` """

    dash_mem: Union[IntVector, FloatVector] = field(default_factory=list)
    """ IntVector: The membrane decay parameters for each neuron ``(N,)`` """

    dash_syn: Union[IntVector, FloatVector] = field(default_factory=list)
    """ IntVector: The synapse decay parameters for each neuron. Either ``(N,)`` if only one synapse is used per neuron, or ``(2N,)`` if two synapses are used for each neuron (i.e. syn2). In this case, elements ``dash_syn[0:1]`` refer to the synapses of neuron ``0``, and so on. """

    dt: Optional[float] = None
    """ float: The ``dt`` time step used for this neuron module """

    @classmethod
    def _convert_from(cls, mod: GraphModule) -> GraphModule:
        if isinstance(mod, cls):
            # - No need to do anything
            return mod

        elif isinstance(mod, LIFNeuronWithSynsRealValue):
            # - Convert from a real-valued LIF neuron
            # - Get a value for `dt` to use in the conversion
            if mod.dt is None:
                raise ValueError(
                    f"Graph module of type {type(mod).__name__} has no `dt` set, so cannot convert time constants when converting to {cls.__name__}."
                )

            # - Convert TCs to dash parameters
            dash_mem = np.log2(np.array(mod.tau_mem) / mod.dt).tolist()
            dash_syn = np.log2(np.array(mod.tau_syn) / mod.dt).flatten().tolist()

            # - Get thresholds
            thresholds = np.array(mod.threshold).tolist()

            # - Build a new neurons module to insert into the graph
            neurons = cls._factory(
                len(mod.input_nodes),
                len(mod.output_nodes),
                mod.name,
                mod.computational_module,
                [],  # Empty list for HW IDs
                thresholds,
                dash_mem,
                dash_syn,
                mod.dt,
            )

            # - Replace the target module and return
            replace_module(mod, neurons)
            return neurons

        elif isinstance(mod, GenericNeurons):
            # - Try to convert as a `GenericNeurons` base class
            if type(mod) != GenericNeurons:
                # - Warn if `mod` is actually some other derived class
                #   We might be missing an explicit conversion rule in this case
                warnings.warn(
                    f"Converting module {mod} as a GenericNeurons module to {cls.__name__} . No explicit conversion rule was found for class {type(mod).__name__}."
                )

            # - Make a new module
            neurons = cls._factory(
                len(mod.input_nodes),
                len(mod.output_nodes),
                mod.name,
            )

            # - Replace the target module
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
    """
    A :py:class:`.graph.GraphModule` encapsulating Xylo v1 hidden neurons
    """

    def __post_init__(self, *args, **kwargs):
        if len(self.input_nodes) != len(self.output_nodes):
            if len(self.input_nodes) != 2 * len(self.output_nodes):
                raise ValueError(
                    "Number of input nodes must be 1* or 2* number of output nodes"
                )

        super().__post_init__(self, *args, **kwargs)


@dataclass(eq=False, repr=False)
class XyloOutputNeurons(XyloNeurons):
    """
    A :py:class:`.graph.GraphModule` encapsulating Xylo V1 output neurons
    """

    def __post_init__(self, *args, **kwargs):
        if len(self.input_nodes) != len(self.output_nodes):
            raise ValueError(
                "Number of input nodes must be equal to number of output nodes"
            )

        super().__post_init__(self, *args, **kwargs)
