"""
Base classes and functionality for graph tracing
"""
import copy
from dataclasses import dataclass, field
from typing import List, Any, TypeVar, Iterable, Hashable, Optional

ArrayLike = Any
Module = Any

__all__ = [
    "GraphNode",
    "GraphModule",
    "GraphModuleBase",
    "GraphHolder",
    "as_GraphHolder",
    "SetList",
]

T = TypeVar("T")


class SetList(List[T]):
    """
    A List class that implements unique adding and appending, and maintains item order

    On construction, only unique items will be retained.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if len(self) > 0:
            old_list = copy.copy(self)
            self.clear()
            [self.add(item) for item in old_list]

    def append(self, __object: T) -> None:
        """
        Add an element to the SetList only if it is not already in the collection

        Args:
            __object: element to add to the SetList
        """
        self.add(__object)

    def add(self, element: Hashable) -> None:
        """
        Add an element to the SetList only if it is not already in the collection

        Args:
            element (Hashable): An element to add to the collection
        """
        if element not in self:
            super().append(element)

    def extend(self, __iterable: Iterable[T]) -> None:
        """
        Add several elements to the SetList, if they are not already in the collection

        Args:
            __iterable (Iterable[Hashable]): An iterable of elements to add to the collection
        """
        [self.add(e) for e in __iterable]


@dataclass(eq=False)
class GraphModuleBase:
    """
    Base class for graph modules

    You should use the "public facing" base classes :py:class:`.GraphModule` and :py:class:`.GraphHolder`.

    See Also:
        For an overview of computational graphs in Rockpool, see :ref:`/advanced/graph_overview.ipynb`.
    """

    input_nodes: SetList["GraphNode"]
    """ SetList[GraphNode]: The input nodes attached to this module """

    output_nodes: SetList["GraphNode"]
    """ SetList[GraphNode]: The output nodes attached to this module """

    name: str
    """ str: An arbitrary name attached to this specific :py:class:`.GraphModule` """

    computational_module: Module
    """ Module: The computational module that acts as the source for this graph module """

    def __post_init__(self, *args, **kwargs):
        # - Ensure node lists are SetLists
        self.input_nodes = SetList(self.input_nodes)
        self.output_nodes = SetList(self.output_nodes)

    def __repr__(self) -> str:
        type_name = type(self).__name__
        name_str = f' "{self.name}"' if len(self.name) > 0 else ""
        return f"{type_name}{name_str} with {len(self.input_nodes)} input nodes -> {len(self.output_nodes)} output nodes"

    def add_input(self, node: "GraphNode") -> None:
        """
        Add a :py:class:`.GraphNode` as an input source to this module

        Args:
            node (GraphNode): The node to add as an input source. This node will be appended after the last current input node.
        """
        self.input_nodes.append(node)

    def add_output(self, node: "GraphNode") -> None:
        """
        Add a :py:class:`.GraphNode` as an output of this module

        Args:
            node (GraphNode): The node to add as an output channel. This node will be appended after the last current output node.
        """
        self.output_nodes.append(node)

    def remove_input(self, node: "GraphNode") -> None:
        """
        Remove a :py:class:`.GraphNode` as an input of this module

        Args:
            node (GraphNode): The node to remove. If this node exists as an input to the module, it will be removed.
        """
        if node in self.input_nodes:
            self.input_nodes.remove(node)

    def remove_output(self, node: "GraphNode") -> None:
        """
        Remove a :py:class:`.GraphNode` as an output of this module

        Args:
            node (GraphNode): The node to remove. If this node exists as an output of the module, it will be removed.
        """
        if node in self.output_nodes:
            self.output_nodes.remove(node)

    def clear_inputs(self) -> None:
        """
        Remove all :py:class:`.GraphNode` s as inputs of this module
        """
        input_nodes = copy.copy(self.input_nodes)
        for i_n in input_nodes:
            self.remove_input(i_n)

    def clear_outputs(self) -> None:
        """
        Remove all :py:class:`.GraphNode` s as outputs of this module
        """
        output_nodes = copy.copy(self.output_nodes)
        for o_n in output_nodes:
            self.remove_output(o_n)

    @classmethod
    def _factory(
        cls,
        size_in: int,
        size_out: int,
        name: str = None,
        computational_module: Optional[Module] = None,
        *args,
        **kwargs,
    ) -> "GraphModuleBase":
        """
        Build a new :py:class:`.GraphModule` or :py:class:`.GraphModule` subclass, with new input and output :py:class:`.GraphNode` s created automatically

        Use this factory method to construct a new :py:class:`.GraphModule` from scratch, which needs new input and output :py:class:`.GraphNode` s created automatically. This helper method will be inherited by new :py:class:`.GraphModule` subclasses, and will act as factory methods also for your custom :py:class:`.GraphModule` subclass.

        Args:
            size_in (int): The number of input :py:class:`.GraphNode` s to create and attach
            size_out (int):  The number of output :py:class:`.GraphNode` s to create and attach
            name (str): An arbitrary name to attach to this :py:class:`.GraphModule`
            computational_module (Module): A rockpool computational module that forms the "generator" of this graph module
            *args, **kwargs: Any additional arguments to pass to the specific subclass constructor

        Returns:
            GraphModule: The newly constructed :py:class:`.GraphModule` or :py:class:`.GraphModule` subclass
        """
        # - Generate nodes
        input_nodes = SetList([GraphNode() for _ in range(size_in)])
        output_nodes = SetList([GraphNode() for _ in range(size_out)])

        # - Build module
        return cls(
            input_nodes, output_nodes, name, computational_module, *args, **kwargs
        )


@dataclass(eq=False)
class GraphNodeBase:
    """
    Base class for GraphNodes
    """

    pass


@dataclass(eq=False, repr=False)
class GraphModule(GraphModuleBase):
    """
    Describe a module of computation in a graph

    :py:class:`.GraphModule` acts as a base class for all modules of computation that exist in a computational graph in Rockpool. It minimally holds a set of input nodes :py:attr:`.input_nodes` and output nodes :py:attr:`.output_nodes` that it is connected to, and which are then connected onward to other :py:class:`.GraphModule` s in the graph.

    You should subclass :py:class:`.GraphModule` to make graph modules that define some specific computation. e.g. a set of weights; a set of spiking neurons with some characteristics. Specific subclasses of :py:class:`.GraphModule` can be converted between each other as part of the device mapping process. e.g. a set of generic LIF neurons can be converted to a HW specific graph module that defines the configuration of some neurons on a device.

    Specific subclasses of :py:class:`.GraphModule` should ideally be "units" of computation, as in the examples above. The exception is :py:class:`.GraphHolder`, which is deigned to encapsulate entire graphs or subgraphs.

    See Also:
        For an overview of computational graphs in Rockpool, see :ref:`/advanced/graph_overview.ipynb`.
    """

    def __post_init__(self, *args, **kwargs):
        """
        Perform any post-initialisation checks that need to be done for this class. You must call `super().__post_init__(*args, **kwargs)` if you override :py:meth:`.__post_init__` in a subclass.
        """
        super().__post_init__(*args, **kwargs)

        # - Attach input and output nodes back to module
        for n in self.input_nodes:
            n.add_sink(self)

        for n in self.output_nodes:
            n.add_source(self)

    @classmethod
    def _convert_from(cls, mod: GraphModuleBase) -> GraphModuleBase:
        """
        Convert another :py:class:`.GraphModule` to a :py:class:`.GraphModule` of this specific subclass

        You should override this method in your subclass, to include conversion rules from other graph module classes to your specific subclass.

        If you do not provide conversion rules to your specific subclass then it will not be possible to map other :py:class:`.GraphModule` subclasses to your subclass.

        Args:
            mod (GraphModule): A :py:class:`.GraphModule` or :py:class:`.GraphModule` subclass object to convert to an object of the specific subclass.

        Returns:
            GraphModule: A converted :py:class:`.GraphModule` subclass object, of the specific subclass on which this method was called.
        """
        raise ValueError(
            f"No conversion rules implemented for the class {cls.__name__}."
        )

    def add_input(self, node: "GraphNode") -> None:
        """
        Add a :py:class:`.GraphNode` as an input source to this module, and connect it

        The new node will be appended after the last current input node. The node will be connected with this :py:class:`.GraphModule` as a sink.

        Args:
            node (GraphNode): The node to add as an input source
        """
        super().add_input(node)
        node.add_sink(self)

    def add_output(self, node: "GraphNode") -> None:
        """
        Add a :py:class:`.GraphNode` as an output of this module, and connect it

        The new node will be appended after the last current output node. The node will be connected with this :py:class:`.GraphModule` as a source.

        Args:
            node (GraphNode): The node to add as an output
        """
        super().add_output(node)
        node.add_source(self)

    def remove_input(self, node: "GraphNode") -> None:
        """
        Remove a :py:class:`.GraphNode` as an input of this module, and disconnect it

        The node will be disconnected from this :py:class:`.GraphModule` as a sink, and will be removed from the module.

        Args:
            node (GraphNode): The node to remove. If this node exists as an input to the module, it will be removed.
        """
        super().remove_input(node)
        node.remove_sink(self)

    def remove_output(self, node: "GraphNode") -> None:
        """
        Remove a :py:class:`.GraphNode` as an output of this module, and disconnect it

        The node will be disconnected from this :py:class:`.GraphModule` as a source, and will be removed from the module.

        Args:
            node (GraphNode): The node to remove. If this node exists as an output to the module, it will be removed.
        """
        super().remove_output(node)
        node.remove_source(self)


@dataclass(eq=False, repr=False)
class GraphHolder(GraphModuleBase):
    """
    A :py:class:`.GraphModule` that encapsulates other graphs

    This module is used to simply encapsulate a graph, and has no computational function. This module can be removed from the graph by wiring up its submodules directly, without modifying the computational structure of the graph.

    :py:class:`.GraphHolder` modules contain only :py:attr:`.input_nodes` and :py:attr:`.input_nodes` attributes that are connected to other modules. A :py:class:`.GraphHolder` module should *never* be the sink or source of a :py:class:`.GraphNode`.

    See Also:
        Use the :py:func:`~.graph.graph_base.as_GraphHolder` helper function to encapsulate another :py:class:`.GraphModule`.

        For an overview of computational graphs in Rockpool, see :ref:`/advanced/graph_overview.ipynb`.
    """

    pass


def as_GraphHolder(g: GraphModule) -> GraphHolder:
    """
    Encapsulate a :py:class:`.GraphModule` inside a :py:class:`.GraphHolder`

    This function takes an existing :py:class:`.GraphModule` and wraps it in a :py:class:`.GraphHolder` module, by using the input and output nodes of the existing module.

    Args:
        g (GraphModule): A :py:class:`.GraphModule` to encapsulate

    Returns:
        GraphHolder: A :py:class:`.GraphHolder` encapsulating `g`
    """
    return GraphHolder(
        input_nodes=g.input_nodes,
        output_nodes=g.output_nodes,
        name=g.name,
        computational_module=None,
    )


@dataclass(eq=False, repr=False)
class GraphNode:
    """
    Describe a node connecting :py:class:`.GraphModule` s

    :py:class:`.GraphNode` s are elements that connect multiple :py:class:`.GraphModule` s. They maintain lists of source and sink modules. Use the methods :py:meth:`.add_source`, :py:meth:`.add_sink`, :py:meth:`.remove_source` and :py:meth:`.remove_sink` to connect and disconnect :py:class:`GraphNode` s.

    See Also:
        For an overview of computational graphs in Rockpool, see :ref:`/advanced/graph_overview.ipynb`.
    """

    source_modules: SetList[GraphModule] = field(default_factory=SetList)
    """ SetList[GraphModule]: The source modules that connect via this :py:class:`.GraphNode` """

    sink_modules: SetList[GraphModule] = field(default_factory=SetList)
    """ SetList[GraphModule]: The sink modules that connect via this :py:class:`.GraphNode` """

    def __post_init__(self, *args, **kwargs):
        # - Ensure node lists are SetLists
        self.source_modules = SetList(self.source_modules)
        self.sink_modules = SetList(self.sink_modules)

    def add_sink(self, sink: GraphModule) -> None:
        """
        Add a :py:class:`.GraphModule` as a sink of this :py:class:`.GraphNode`

        Args:
            sink (GraphModule): The module to add to this node

        Raises:
            ValueError: If `sink` is a :py:class:`.GraphHolder`.
        """
        if isinstance(sink, GraphHolder):
            raise ValueError(
                f"A `GraphHolder` object may not be added as a node sink. I was given {sink}."
            )

        self.sink_modules.add(sink)

    def add_source(self, source: GraphModule) -> None:
        """
        Add a :py:class:`.GraphModule` as a source of this :py:class:`.GraphNode`

        Args:
            source (GraphModule): The module to add to this node

        Raises:
            ValueError: If `source` is a :py:class:`.GraphHolder`.
        """
        if isinstance(source, GraphHolder):
            raise ValueError(
                f"A `GraphHolder` object may not be added as a node source. I was given {source}."
            )

        self.source_modules.add(source)

    def remove_sink(self, sink: GraphModule) -> None:
        """
        Remove a :py:class:`.GraphModule` as a sink of this :py:class:`.GraphNode`

        If `sink` is a sink of this :py:class:`.GraphNode`, it will be removed.

        Args:
            sink (GraphModule): The module to remove from this node
        """
        if sink in self.sink_modules:
            self.sink_modules.remove(sink)

    def remove_source(self, source: GraphModule) -> None:
        """
        Remove a :py:class:`.GraphModule` as a source of this :py:class:`.GraphNode`

        If `source` is a source of this :py:class:`.GraphNode`, it will be removed.

        Args:
            source (GraphModule): The module to remove from this node
        """
        if source in self.source_modules:
            self.source_modules.remove(source)

    def __repr__(self) -> str:
        type_name = type(self).__name__

        if self.source_modules is None:
            input_str = "no inputs"
        else:
            input_str = f"{len(self.source_modules)} source modules"

        if self.sink_modules is None:
            output_str = "no outputs"
        else:
            output_str = f"{len(self.sink_modules)} sink modules"

        return f"{type_name} {id(self)} with {input_str} and {output_str}"
