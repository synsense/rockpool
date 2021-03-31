"""
Cimulator-backed module compatible with Pollen. Requires Cimulator
"""

# - Check that Cimulator is installed
from importlib import util

if util.find_spec("cimulator") is None:
    raise ModuleNotFoundError(
        "'Cimulator' not found. Modules that rely on Cimulator will not be available."
    )

# - Rockpool imports
from rockpool.nn.modules import Module
from rockpool.parameters import Parameter, State, SimulationParameter

# - Import Cimulator
from cimulator.pollen import Synapse, PollenLayer

# - Numpy
import numpy as np

# - Typing
from typing import Optional, Union

# - Define exports
__all__ = ["PollenCim"]


class PollenCim(Module):
    """
    A Module simulating a digital SNN on Pollen, using Cimulator as a back-end
    """

    __create_key = object()
    """ Private key to ensure factory creation """

    def __init__(
        self,
        create_key,
        config: "PollenConfiguration",
        shape: tuple = (16, 1000, 8),
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Warnings:
            Use the factory methods :py:meth:`.PollenCim.from_config` and :py:meth:`PollenCim.from_specfication` to construct a :py:class:`.PollenCim` module.

        Args:
            create_key:
            shape:
            dt:
            *args:
            **kwargs:
        """
        # - Check that we are creating the object using a factory function
        if create_key is not PollenCim.__create_key:
            raise NotImplementedError(
                "PollenCim may only be instantiated using factory methods `from_config` or `from_weights`."
            )

        # - Initialise the superclass
        super().__init__(
            dt=dt,
            shape=shape,
            spiking_input=True,
            spiking_output=True,
            *args,
            **kwargs,
        )

        # - Store the configuration
        self.config: Union["PollenConfiguration", Parameter] = Parameter(config)
        """ (PollenConfiguration) Configuration of the Pollen module """

    @classmethod
    def from_config(cls, config: "PollenConfiguration", dt: float = 1e-3):
        """
        Cimulator based layer to simulate the Pollen hardware.

        Parameters:
        dt: float
            Timestep for simulation, in seconds. Default: 1ms
        config: PollenConfiguration
            ``samna.pollen.Configuration`` object to specify all parameters. See samna documentation for details.

        """
        # - Instantiate the class
        mod = cls(create_key=cls.__create_key, config=config, dt=dt)

        synapses_in = []
        for pre, w_pre in enumerate(config.input_expansion.weights):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(Synapse(post, 0, w_pre[post]))

            if config.synapse2_enable:
                w2_pre = config.input_expansion.syn2_weights[pre]
                for post in np.where(w2_pre)[0]:
                    tmp.append(Synapse(post, 1, w2_pre[post]))

            synapses_in.append(tmp)

        synapses_rec = []
        for pre, w_pre in enumerate(config.reservoir.weights):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(Synapse(post, 0, w_pre[post]))

            if config.synapse2_enable:
                w2_pre = config.reservoir.syn2_weights[pre]
                for post in np.where(w2_pre)[0]:
                    tmp.append(Synapse(post, 1, w2_pre[post]))

            synapses_rec.append(tmp)

        synapses_out = []
        for pre, w_pre in enumerate(config.readout.weights):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(Synapse(post, 0, w_pre[post]))
            synapses_out.append(tmp)

        threshold = []
        dash_syn = []
        dash_mem = []
        aliases = []

        for neuron in config.reservoir.neurons:
            if neuron.alias_target:
                aliases.append([neuron.alias_target])
            else:
                aliases.append([])
            threshold.append(neuron.threshold)
            dash_mem.append(neuron.v_mem_decay)
            dash_syn.append([neuron.i_syn_decay, neuron.i_syn2_decay])

        threshold_out = []
        dash_syn_out = []
        dash_mem_out = []

        for neuron in config.readout.neurons:
            threshold_out.append(neuron.threshold)
            dash_mem_out.append(neuron.v_mem_decay)
            dash_syn_out.append([neuron.i_syn_decay])

        weight_shift_inp = config.input_expansion.weight_bit_shift
        weight_shift_rec = config.reservoir.weight_bit_shift
        weight_shift_out = config.readout.weight_bit_shift

        mod._pollen_layer = PollenLayer(
            synapses_in=synapses_in,
            synapses_rec=synapses_rec,
            synapses_out=synapses_out,
            aliases=aliases,
            threshold=threshold,
            threshold_out=threshold_out,
            weight_shift_inp=weight_shift_inp,
            weight_shift_rec=weight_shift_rec,
            weight_shift_out=weight_shift_out,
            dash_mem=dash_mem,
            dash_mem_out=dash_mem_out,
            dash_syns=dash_syn,
            dash_syns_out=dash_syn_out,
            name="PollenCim_PollenLayer",
        )

        return mod

    @classmethod
    def from_specification(
        cls,
        weights_in: np.ndarray,
        weights_rec: np.ndarray,
        weights_out: np.ndarray,
        dash_mem: np.ndarray,
        dash_mem_out: np.ndarray,
        dash_syn: np.ndarray,
        dash_syn_out: np.ndarray,
        threshold: np.ndarray,
        threshold_out: np.ndarray,
        weight_shift: int = 0,
        weight_shift_out: int = 0,
        aliases: Optional[list] = None,
    ) -> "PollenCim":
        """
        Instantiate a :py:class:`.PollenCim` module from a full set of parameters

        Args:
            weights_in (np.ndarray): An
            weights_rec (np.ndarray):
            weights_out (np.ndarray):
            dash_mem (np.ndarray):
            dash_mem_out (np.ndarray):
            dash_syn (np.ndarray):
            dash_syn_out (np.ndarray):
            threshold (np.ndarray):
            threshold_out (np.ndarray):
            weight_shift (int):
            weight_shift_out (int):
            aliases (Optional[list]):

        Returns: :py:class:`.PollenCim`: A :py:class:`.TimedModule` that

        """
        from rockpool.devices.pollen import config_from_specification

        # - Convert specification to pollen configuration
        config, _ = config_from_specification(
            weights_in,
            weights_rec,
            weights_out,
            dash_mem,
            dash_mem_out,
            dash_syn,
            dash_syn_out,
            threshold,
            threshold_out,
            weight_shift,
            weight_shift_out,
            aliases,
        )

        # - Instantiate module from config
        return cls.from_config(config)

    def evolve(
        self, input_raster: np.ndarray = None, record: bool = False, *args, **kwargs,
    ):
        # - Evolve using the pollen layer
        output = self._pollen_layer.evolve(input_raster)

        # - Build the recording dictionary
        if not record:
            recording = {}
        else:
            recording = {
                "vmem": self._pollen_layer.rec_v_mem,
                "isyn": self._pollen_layer.rec_i_syn,
                "isyn2": self._pollen_layer.rec_i_syn2,
                "spikes": self._pollen_layer.rec_recurrent_spikes,
                "vmem_out": self._pollen_layer.rec_v_mem_out,
                "isyn_out": self._pollen_layer.rec_i_syn_out,
            }

        # - Return output, state and recording dictionary
        return output, {}, recording

    def reset_state(self) -> "PollenCim":
        """ Reset the state of this module. """
        self._pollen_layer.reset_all()
        return self
