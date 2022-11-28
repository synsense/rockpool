"""
Dynap-SE graph graph mapper package

- Create a graph using the :py:meth:`~.graph.GraphModule.as_graph` API
- Call :py:func:`.mapper`

Note : Existing modules are reconstructed considering consistency with Xylo support.


Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022
"""
from __future__ import annotations
from typing import Dict, Optional

import numpy as np
from copy import deepcopy
from dataclasses import dataclass

from rockpool.typehints import FloatVector

from rockpool.nn.modules.module import ModuleBase
from rockpool.nn.modules import LinearJax
from rockpool.nn.combinators import Sequential

from rockpool.graph import GraphModule, GraphHolder
from rockpool.graph.utils import bag_graph
from rockpool.graph.graph_modules import LinearWeights

from rockpool.devices.dynapse.dynapsim import DynapSim
from rockpool.devices.dynapse.mapping import DynapseNeurons
from rockpool.devices.dynapse.config.transform import transformer, recurrent_modules
from rockpool.devices.dynapse.typehints import DRCError


__all__ = ["DynapseGraphContainer", "mapper"]


@dataclass
class DynapseGraphContainer:
    """
    DynapseGraphContainer is a helper for mapping pipeline. It checks if the given DynapSim graph is properly constructed.
    Stores the simulator, input weight and recurrent weight graph in a structured way

    :Parameters:
    :param simulator: the core simulator graph which contains the parameter currents
    :type simulator: DynapseNeurons
    :param input: the linear input layer graph which contains input weight matrix, does not need to be defined, can be None
    :type input: Optional[LinearWeights]
    :param recurrent: the recurrent weight graph which contatins recurrent weight matrix, does not need to be defined, can be None
    :type recurrent: Optional[LinearWeights]
    """

    simulator: DynapseNeurons
    input: Optional[LinearWeights]
    recurrent: Optional[LinearWeights]

    @classmethod
    def from_graph_holder(cls, graph: GraphHolder) -> DynapseGraphContainer:
        """
        from_graph_holder constructs a ``DynapseGraphContainer`` object from a ``GraphHolder`` object which holds a proper DynapSim graph

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
        nodes, modules = bag_graph(graph)
        rec_mod = recurrent_modules(modules)

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

    def get_sequential(self) -> ModuleBase:
        """
        get_sequential restores the stored DynapSim computational graph as a ``JaxSequential`` combinator.

        :return: a computational module restored from computational graph stored
        :rtype: ModuleBase
        """

        in_layer = LinearJax.from_graph(self.input) if self.input is not None else None
        dynapsim_layer = DynapSim.from_graph(self.simulator, self.recurrent)

        if in_layer is None:
            mod = Sequential(dynapsim_layer)
        else:
            mod = Sequential(in_layer, dynapsim_layer)

        return mod

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


def mapper(graph: GraphModule, in_place=False) -> Dict[str, float]:
    """
    mapper maps a computational graph onto Dynap-SE2 architecture.

    It requires post-processing steps such as
        * weight_quantization (float -> 4-bit)
        * parameter_selection (Isyn -> I_ampa, Igaba, Inmda, Ishunt)
        * parameter_quantization (current value -> coarse-fine : 1e-8 -> (3,75))
        * parameter_clustering (allocates seperate cores for different groups)
        * config (create a samna config object to deploy everything to hardware)

    :param graph: Any graph(constraint) aimed to be deployed to Dynap-SE2
    :type graph: GraphModule
    :return: a specification object which can be used to create a config object
    :rtype: Dict[str, float]
    """

    try:
        wrapper = DynapseGraphContainer.from_graph_holder(graph)
    except:
        graph = transformer(deepcopy(graph)) if not in_place else transformer(graph)
        wrapper = DynapseGraphContainer.from_graph_holder(graph)

    specs = {
        "mapped_graph": graph,
        "weights_in": wrapper.w_in,
        "weights_rec": wrapper.w_rec,
        "Iscale": wrapper.Iscale,
    }

    specs.update(wrapper.current_dict)

    return specs
