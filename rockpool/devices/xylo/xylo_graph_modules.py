from rockpool.graph import GenericNeurons, GraphModule, replace_module

from typing import List

from dataclasses import dataclass, field

__all__ = ["XyloNeurons", "XyloHiddenNeurons", "XyloOutputNeurons"]


@dataclass(eq=False, repr=False)
class XyloNeurons(GenericNeurons):
    hw_ids: List[int] = field(default_factory=list)
    thresholds: List[int] = field(default_factory=list)
    tau_mem: List[int] = field(default_factory=list)
    tau_syn: List[int] = field(default_factory=list)

    @classmethod
    def _swap(cls, mod: GraphModule) -> GraphModule:
        if isinstance(mod, cls):
            # - No need to do anything
            return mod

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
                f"Graph module of type {type(mod).__name__} cannot be swapped with a {cls.__name__}"
            )


@dataclass(eq=False, repr=False)
class XyloHiddenNeurons(XyloNeurons):
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(self, *args, **kwargs)

        if len(self.input_nodes) != len(self.output_nodes):
            if len(self.input_nodes) != 2 * len(self.output_nodes):
                raise ValueError(
                    "Number of input nodes must be 1* or 2* number of output nodes"
                )


@dataclass(eq=False, repr=False)
class XyloOutputNeurons(XyloNeurons):
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(self, *args, **kwargs)

        if len(self.input_nodes) != len(self.output_nodes):
            raise ValueError(
                "Number of input nodes must be equal to number of output nodes"
            )
