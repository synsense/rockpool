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
            shape=shape,
            spiking_input=True,
            spiking_output=True,
            *args,
            **kwargs,
        )

        # - Store the configuration
        self.config: Union["PollenConfiguration", Parameter] = Parameter(
            init_func=lambda _: config
        )
        """ (PollenConfiguration) Configuration of the Pollen module """

        # - Store the dt
        self.dt: Union[float, SimulationParameter] = SimulationParameter(dt)
        """ (float) Simulation time-step for this module """

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

        # - Make a storage object for the extracted configuration
        class _(object):
            pass

        _cim_params = _()

        # - Convert input weights to Synapse objects
        _cim_params.synapses_in = []
        for pre, w_pre in enumerate(config.input_expansion.weights):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(Synapse(post, 0, w_pre[post]))

            if config.synapse2_enable:
                w2_pre = config.input_expansion.syn2_weights[pre]
                for post in np.where(w2_pre)[0]:
                    tmp.append(Synapse(post, 1, w2_pre[post]))

            _cim_params.synapses_in.append(tmp)

        # - Convert recurrent weights to Synapse objects
        _cim_params.synapses_rec = []
        for pre, w_pre in enumerate(config.reservoir.weights):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(Synapse(post, 0, w_pre[post]))

            if config.synapse2_enable:
                w2_pre = config.reservoir.syn2_weights[pre]
                for post in np.where(w2_pre)[0]:
                    tmp.append(Synapse(post, 1, w2_pre[post]))

            _cim_params.synapses_rec.append(tmp)

        # - Convert output weights to Synapse objects
        _cim_params.synapses_out = []
        for pre, w_pre in enumerate(config.readout.weights):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(Synapse(post, 0, w_pre[post]))
            _cim_params.synapses_out.append(tmp)

        # - Configure reservoir neurons
        _cim_params.threshold = []
        _cim_params.dash_syn = []
        _cim_params.dash_mem = []
        _cim_params.aliases = []

        for neuron in config.reservoir.neurons:
            if neuron.alias_target:
                _cim_params.aliases.append([neuron.alias_target])
            else:
                _cim_params.aliases.append([])
            _cim_params.threshold.append(neuron.threshold)
            _cim_params.dash_mem.append(neuron.v_mem_decay)
            _cim_params.dash_syn.append([neuron.i_syn_decay, neuron.i_syn2_decay])

        # - Configure readout neurons
        _cim_params.threshold_out = []
        _cim_params.dash_syn_out = []
        _cim_params.dash_mem_out = []

        for neuron in config.readout.neurons:
            _cim_params.threshold_out.append(neuron.threshold)
            _cim_params.dash_mem_out.append(neuron.v_mem_decay)
            _cim_params.dash_syn_out.append([neuron.i_syn_decay])

        _cim_params.weight_shift_inp = config.input_expansion.weight_bit_shift
        _cim_params.weight_shift_rec = config.reservoir.weight_bit_shift
        _cim_params.weight_shift_out = config.readout.weight_bit_shift

        # - Instantiate a Pollen Cimulation layer
        mod._pollen_layer = PollenLayer(
            synapses_in=_cim_params.synapses_in,
            synapses_rec=_cim_params.synapses_rec,
            synapses_out=_cim_params.synapses_out,
            aliases=_cim_params.aliases,
            threshold=_cim_params.threshold,
            threshold_out=_cim_params.threshold_out,
            weight_shift_inp=_cim_params.weight_shift_inp,
            weight_shift_rec=_cim_params.weight_shift_rec,
            weight_shift_out=_cim_params.weight_shift_out,
            dash_mem=_cim_params.dash_mem,
            dash_mem_out=_cim_params.dash_mem_out,
            dash_syns=_cim_params.dash_syn,
            dash_syns_out=_cim_params.dash_syn_out,
            name="PollenCim_PollenLayer",
        )

        # - Store parameters and return
        mod._cim_params = _cim_params
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
        self,
        input_raster: np.ndarray = None,
        record: bool = False,
        *args,
        **kwargs,
    ):
        # - Evolve using the pollen layer
        output = np.array(self._pollen_layer.evolve(input_raster))

        # - Build the recording dictionary
        if not record:
            recording = {}
        else:
            recording = {
                "vmem": np.array(self._pollen_layer.rec_v_mem).T,
                "isyn": np.array(self._pollen_layer.rec_i_syn).T,
                "isyn2": np.array(self._pollen_layer.rec_i_syn2).T,
                "spikes": np.array(self._pollen_layer.rec_recurrent_spikes),
                "vmem_out": np.array(self._pollen_layer.rec_v_mem_out).T,
                "isyn_out": np.array(self._pollen_layer.rec_i_syn_out).T,
            }

        # - Return output, state and recording dictionary
        return output, {}, recording

    def reset_state(self) -> "PollenCim":
        """ Reset the state of this module. """
        self._pollen_layer.reset_all()
        return self
