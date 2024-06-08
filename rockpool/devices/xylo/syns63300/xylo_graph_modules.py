"""
Xylo graph modules for use with tracing and mapping
"""

from rockpool.devices.xylo.syns61201 import Xylo2Neurons

from dataclasses import dataclass, field

__all__ = ["XyloIMUNeurons", "XyloIMUHiddenNeurons", "XyloIMUOutputNeurons"]


class XyloIMUNeurons(Xylo2Neurons):
    """
    Base class for all Xylo graph module classes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __post_init__(self, *args, **kwargs):
        if len(self.input_nodes) != len(self.output_nodes):
            raise ValueError(
                "Number of input nodes must be equal to number of output nodes"
            )

        super().__post_init__(self, *args, **kwargs)


@dataclass(eq=False, repr=False)
class XyloIMUHiddenNeurons(XyloIMUNeurons):
    """
    A :py:class:`.graph.GraphModule` encapsulating Xylo IMU hidden neurons
    """

    pass


@dataclass(eq=False, repr=False)
class XyloIMUOutputNeurons(XyloIMUNeurons):
    """
    A :py:class:`.graph.GraphModule` encapsulating Xylo IMU output neurons
    """

    pass
