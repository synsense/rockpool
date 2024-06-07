"""
Dynap-SE2 graph container implementation helping the operation of `devices.dynapse.mapper` function

* Non User Facing *
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np
from dataclasses import dataclass

from rockpool.typehints import FloatVector

from rockpool.graph import GraphHolder
from rockpool.graph.utils import find_recurrent_modules
from rockpool.graph.graph_modules import LinearWeights
from rockpool.typehints import DRCError

from .graph import DynapseNeurons

__all__ = ["DynapseGraphContainer"]


@dataclass
class DynapseGraphContainer:
    """
    DynapseGraphContainer is a helper for mapping pipeline. It checks if the given `devices.dynapse.DynapSim` graph is properly constructed.
    Stores the simulator, input weight and recurrent weight graph in a structured way
    """

    simulator: DynapseNeurons
    """the core simulator graph which contains the parameter currents"""

    input: Optional[LinearWeights]
    """the linear input layer graph which contains input weight matrix, does not need to be defined, can be None"""

    recurrent: Optional[LinearWeights]
    """the recurrent weight graph which contatins recurrent weight matrix, does not need to be defined, can be None"""

    @classmethod
    def from_graph_holder(cls, graph: GraphHolder) -> DynapseGraphContainer:
        """
        from_graph_holder constructs a `DynapseGraphContainer` object from a `GraphHolder` object which holds a proper DynapSim graph

        :param graph: a proper computational DynapSim graph object
        :type graph: GraphHolder
        :raises DRCError: Illegal DynapSim Graph! More than 3 connected modules are not expected!
        :raises DRCError: No module found in the given graph holder!
        :raises DRCError: Single module should be ``DynapseNeurons``
        :raises DRCError: First position : ``LinearWeights`` expected
        :raises DRCError: Second Position : ``DynapseNeurons`` expected!
        :raises DRCError: Full graph position 1 (input layer) : ``LinearWeights`` expected!
        :raises DRCError: Full graph position 2 (rec layer) : ``LinearWeights`` expected!
        :raises DRCError: Full graph position 3 (simulation layer) : ``DynapseNeurons`` expected!
        :return: a structured container which holds simulator, input and recurrent graphs seperately
        :rtype: DynapseGraphContainer
        """
        modules, rec_mod = find_recurrent_modules(graph)

        if len(modules) > 3:
            raise DRCError(
                "Illegal DynapSim Graph! More than 3 connected modules are not expected!"
            )

        if len(modules) == 0:
            raise DRCError("No module found in the given graph holder!")

        # Only the FFWD simulator
        if len(modules) == 1:
            if not isinstance(modules[0], DynapseNeurons):
                raise DRCError("Single module should be ``DynapseNeurons``")

            return cls(simulator=modules[0], input=None, recurrent=None)

        # Recurrent simulator or FFWD simulator with input weights
        if len(modules) == 2:
            if not isinstance(modules[0], LinearWeights):
                raise DRCError("First position : ``LinearWeights`` expected")

            if not isinstance(modules[1], DynapseNeurons):
                raise DRCError("Second Position : ``DynapseNeurons`` expected!")

            # FFWD or REC?
            if modules[0] in rec_mod:
                return cls(simulator=modules[1], input=None, recurrent=modules[0])
            else:
                return cls(simulator=modules[1], input=modules[0], recurrent=None)

        # Input weights + Recurrent Simulator
        if len(modules) == 3:
            if not isinstance(modules[0], LinearWeights):
                raise DRCError(
                    "Full graph position 1 (input layer) : ``LinearWeights`` expected!"
                )

            if not isinstance(modules[1], LinearWeights):
                raise DRCError(
                    "Full graph position 2 (rec layer) : ``LinearWeights`` expected!"
                )

            if not isinstance(modules[2], DynapseNeurons):
                raise DRCError(
                    "Full graph position 3 (simulation layer) : ``DynapseNeurons`` expected!"
                )

            return cls(simulator=modules[2], input=modules[0], recurrent=modules[1])

    @property
    def w_in(self) -> Optional[FloatVector]:
        """w_in returns the input weights stored in input weight graph"""
        return self.input.weights if self.input is not None else None

    @property
    def w_rec(self) -> Optional[FloatVector]:
        """w_rec returns the recurrent weights stored in recurrent weight graph"""
        return self.recurrent.weights if self.recurrent is not None else None

    @property
    def current_dict(self) -> Dict[str, FloatVector]:
        """current_dict returns all the current values provided in the core simulator graph as a dictionary"""

        def __zero_check__(vector: FloatVector) -> FloatVector:
            if (np.array(vector) < 0).any():
                raise ValueError("property should be non-negative!")
            return vector

        currents = self.simulator.get_full()
        currents = {k: __zero_check__(v) for k, v in currents.items()}
        return currents

    @property
    def Iscale(self) -> float:
        """Iscale returns the Iscale current stored in the core simulator graph"""
        return self.simulator.Iscale
