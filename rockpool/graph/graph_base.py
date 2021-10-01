from dataclasses import dataclass, field
from typing import List, Any

ArrayLike = Any

__all__ = ["GraphNode", "GraphModule"]


@dataclass(eq=False)
class GraphModuleBase:
    pass


@dataclass(eq=False)
class GraphNodeBase:
    pass


@dataclass(eq=False, repr=False)
class GraphNode:
    source_modules: List["GraphModule"] = field(default_factory=list)
    sink_modules: List["GraphModule"] = field(default_factory=list)

    def add_target(self, target: "GraphModule"):
        if self.sink_modules is None:
            self.sink_modules = [target]
        else:
            self.sink_modules.append(target)

    def add_source(self, source: "GraphModule"):
        if self.source_modules is None:
            self.source_modules = [source]
        else:
            self.source_modules.append(source)

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
    input_nodes: List[GraphNode]
    output_nodes: List[GraphNode]
    name: str

    def __post_init__(self, *args, **kwargs):
        pass

    def __repr__(self):
        type_name = type(self).__name__
        return f'{type_name} "{self.name}" with {len(self.input_nodes)} input nodes -> {len(self.output_nodes)} output nodes'

    @classmethod
    def _factory(cls, size_in: int, size_out: int, name: str = None, *args, **kwargs):
        # - Generate nodes
        input_nodes = [GraphNode() for _ in range(size_in)]
        output_nodes = [GraphNode() for _ in range(size_out)]

        # - Build module
        return cls(input_nodes, output_nodes, name, *args, **kwargs)

    @classmethod
    def _swap(cls, mod: GraphModuleBase) -> GraphModuleBase:
        raise ValueError("No swapping rules implemented")
