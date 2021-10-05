from dataclasses import dataclass, field
from typing import List, Any, TypeVar, Iterable

ArrayLike = Any

__all__ = ["GraphNode", "GraphModule", "GraphHolder", "graph", "SetList"]

T = TypeVar("T")


class SetList(List[T]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if len(self) > 0:
            unique_list = SetList()
            [unique_list._add_unique(item) for item in self]

    def _add_unique(self, element):
        if element not in self:
            self.append(element)

    def _append_unique(self, elements: Iterable):
        [self._add_unique(e) for e in elements]


@dataclass(eq=False)
class GraphModuleBase:
    pass


@dataclass(eq=False)
class GraphNodeBase:
    pass


@dataclass(eq=False, repr=False)
class GraphNode:
    source_modules: SetList["GraphModule"] = field(default_factory=SetList)
    sink_modules: SetList["GraphModule"] = field(default_factory=SetList)

    def add_sink(self, target: "GraphModule"):
        self.sink_modules._add_unique(target)

    def add_source(self, source: "GraphModule"):
        self.source_modules._add_unique(source)

    def remove_sink(self, target: "GraphModule"):
        if target in self.sink_modules:
            self.sink_modules.remove(target)

    def remove_source(self, source: "GraphModule"):
        if source in self.source_modules:
            self.source_modules.remove(source)

    def __repr__(self):
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


@dataclass(eq=False, repr=False)
class GraphModule(GraphModuleBase):
    input_nodes: SetList[GraphNode]
    output_nodes: SetList[GraphNode]
    name: str

    def __post_init__(self, *args, **kwargs):
        pass

    def __repr__(self):
        type_name = type(self).__name__
        return f'{type_name} "{self.name}" with {len(self.input_nodes)} input nodes -> {len(self.output_nodes)} output nodes'

    def add_input(self, node: GraphNode):
        self.input_nodes._add_unique(node)

    def add_output(self, node: GraphNode):
        self.output_nodes._add_unique(node)

    def remove_input(self, node: GraphNode):
        if node in self.input_nodes:
            self.input_nodes.remove(node)

    def remove_output(self, node: GraphNode):
        if node in self.output_nodes:
            self.output_nodes.remove(node)

    @classmethod
    def _factory(cls, size_in: int, size_out: int, name: str = None, *args, **kwargs):
        # - Generate nodes
        input_nodes = SetList([GraphNode() for _ in range(size_in)])
        output_nodes = SetList([GraphNode() for _ in range(size_out)])

        # - Build module
        return cls(input_nodes, output_nodes, name, *args, **kwargs)

    @classmethod
    def _convert_from(cls, mod: GraphModuleBase) -> GraphModuleBase:
        raise ValueError(
            f"No conversion rules implemented for the class {cls.__name__}."
        )


@dataclass(eq=False, repr=False)
class GraphHolder(GraphModule):
    pass


def graph(g: GraphModule):
    return GraphHolder(
        input_nodes=g.input_nodes,
        output_nodes=g.output_nodes,
        name=g.name,
    )
