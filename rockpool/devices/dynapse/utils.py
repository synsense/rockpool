"""
Utility functions for Dynap-SE1/SE2 simulator

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
23/07/2021
"""

import numpy as np
from rockpool.devices.dynapse.dynapse1_jax import DynapSE1Jax
from rockpool.timeseries import TSEvent, TSContinuous
from rockpool.devices.dynapse.router import Router

from typing import (
    Callable,
    Optional,
    Union,
    Tuple,
    Generator,
    List,
    Dict,
    Any,
)

import jax.numpy as jnp

from rockpool.parameters import Parameter, State, SimulationParameter
from rockpool.typehints import JP_ndarray, P_float
from rockpool.devices.dynapse.biasgen import BiasGen

import matplotlib.pyplot as plt

_PANDAS_AVAILABLE = True

try:
    import pandas as pd

except ModuleNotFoundError as e:
    pd = Any
    print(
        e,
        "\nDevice vs. Simulation comparison dataframes cannot be generated!",
    )
    _PANDAS_AVAILABLE = False


ArrayLike = Union[np.ndarray, List, Tuple]


def spike_to_pulse(
    input_spike: Union[TSEvent, np.ndarray],
    dt: float,
    pulse_width: float,
    amplitude: float,
    name: Optional[str] = "$V_{in}$",
) -> TSContinuous:
    """
    spike_to_pulse converts a discrete multi-channel spike train to continuous multi-channel pulse train signal.
    The function both accepts a `TSEvent` object or simple numpy array as input and produce a `TSContinuos` object.

    :param input_spike: multi-channel spike train with shape `TxN`
    :type input_spike: Union[TSEvent, np.ndarray]
    :param dt: duration of single time step in seconds
    :type dt: float
    :param pulse_width: width of a single pulse in seconds
    :type pulse_width: float
    :param amplitude: the amplitude of the pulse in volts
    :type amplitude: float
    :param name: the name of the multichannel pulse train, defaults to "$V_{in}$"
    :type name: Optional[str], optional
    :return: multi-channel pulse train, with shape `TxN`
    :rtype: TSContinuous
    """

    def channel_iterator() -> Tuple[Generator, int]:
        """
        channel_iterator implements an iterator to go over channels of
        given TSEvent object or the np.ndarray. In this contex, channel
        means a neuron firing and/or the second dimension of the array.

        :raises TypeError: Input spike is neither a TSEvent nor a np.ndarray instance
        :return: channel, steps
            channel: generator object to traverse single neuron spike trains
            steps: number of discrete timesteps
        :rtype: Tuple[Generator, int]
        """

        # TSEvent and np.ndarray are both accepted, actions are different
        if isinstance(input_spike, TSEvent):

            def event_channel() -> np.ndarray:
                """
                event_channel [summary]

                :yield: one dimensional boolean matrix with ``True`` indicating presence of events for each channel.
                :rtype: np.ndarray
                """
                event_raster = input_spike.raster(dt)
                yield from event_raster.T

            steps = int(np.round(input_spike.duration / dt))
            return event_channel(), steps

        elif isinstance(input_spike, np.ndarray):

            def array_channel() -> np.ndarray:
                """
                array_channel [summary]

                :yield: one dimensional boolean matrix with ``True`` indicating presence of events for each channel.
                :rtype: np.ndarray
                """
                yield from input_spike.T

            steps = input_spike.shape[0]
            return array_channel(), steps

        else:
            raise TypeError(
                "Input spike can be either a TSEvent or a numpy array instance!"
            )

    if pulse_width < dt:
        raise ValueError(
            f"Pulse width:{pulse_width:.1e} must be greater than or equal to dt:{dt:.1e}"
        )

    # Get the channel iterator and number of timesteps represented in one channel
    channels, signal_steps = channel_iterator()
    pulse_signal = np.empty((signal_steps, 0))
    pulse_steps = int(np.round(pulse_width / dt))

    # 1D convolution kernel
    kernel = amplitude * np.ones(pulse_steps)

    for spike_train in channels:
        pulse_train = np.convolve(spike_train, kernel, mode="full")
        pulse_train = pulse_train[:signal_steps]
        pulse_train = np.expand_dims(pulse_train, 1)
        pulse_signal = np.hstack((pulse_signal, pulse_train))

    pulse_signal = np.clip(pulse_signal, 0, amplitude)
    pulse_signal = TSContinuous.from_clocked(pulse_signal, dt=dt, name=name)

    return pulse_signal


def custom_spike_train(
    times: ArrayLike,
    channels: Optional[ArrayLike],
    duration: float,
    name: Optional[str] = "Input Spikes",
) -> TSEvent:
    """
    custom_spike_train Generate a custom spike train given exact spike times

    :param times: ``Tx1`` vector of exact spike times
    :type times: ArrayLike
    :param channels: ``Tx1`` vector of spike channels. All events belongs to channels 0 if None
    :type channels: Optional[ArrayLike]
    :param duration: The simulation duration in seconds
    :type duration: float
    :param name: The name of the resulting TSEvent object, defaults to "Input Spikes"
    :type name: Optional[str], optional
    :return: custom generated discrete spike train
    :rtype: TSEvent
    """

    input_sp_ts = TSEvent(
        times=times, channels=channels, t_start=0, t_stop=duration, name=name
    )
    return input_sp_ts


def random_spike_train(
    duration: float,
    n_channels: int,
    rate: float,
    dt: float = 1e-3,
    name: Optional[str] = "Input Spikes",
    channel_labels: Optional[ArrayLike] = None,
) -> Union[TSEvent, Tuple[TSEvent, Dict[int, np.uint16]]]:
    """
    random_spike_train generate a Poisson frozen random spike train and
    Dynap-SE1 compatible virtual universal neuron IDs respective to channels if demanded

    :param duration: The simulation duration in seconds
    :type duration: float
    :param channels: Number of channels, or number of neurons
    :type channels: int
    :param rate: The spiking rate in Hertz(1/s)
    :type rate: float
    :param dt: The time step for the forward-Euler ODE solver, defaults to 1e-3
    :type dt: float, optional
    :param name: The name of the resulting TSEvent object, defaults to "Input Spikes"
    :type name: Optional[str], optional
    :param channel_labels: a list of universal neuron keys to assign channels, defaults to None
    :type channel_labels: Optional[ArrayLike], optional
    :param return_channel_UID: return channels UIDs together with TSEvent spike train or not, defaults to False
    :type return_channel_UID: bool, optional
    :raises ValueError: No spike generated due to low firing rate or very short simulation time]
    :raises ValueError: Channels UID list must have the same number of elements with the number of channels
    :raises ValueError: Duplicate elements found in channelUID list. UID's should be unique!
    :raises ValueError: Illegal virtual neuron UID! It should be less than 1024
    :raises ValueError: Illegal UID ! It should be bigger than 0
    :return: input_sp_ts, channel_UID_dict
        :input_sp_ts: randomly generated discrete spike train
        :channel_UID_dict: a dictionary mapping the channels indexes to neruon UIDs
    :rtype: Union[TSEvent, Tuple[TSEvent, Dict[int, np.uint16]]]

    [] TODO: parametrize max UID = 1024
    """

    steps = int(np.round(duration / dt))
    input_sp_raster = np.random.poisson(rate / steps, (steps, n_channels))
    if not any(input_sp_raster.flatten()):
        raise ValueError(
            "No spike generated at all due to low firing rate or short simulation time duration!"
        )
    input_sp_ts = TSEvent.from_raster(input_sp_raster, name=name, periodic=True, dt=dt)

    if channel_labels is not None:

        channel_UID = list(map(lambda key: Router.get_UID(*key), channel_labels))

        if len(channel_UID) != input_sp_ts.num_channels:
            raise ValueError(
                "Channels UID list must have the same number of elements with the number of channels"
            )
        channel_UID = np.array(channel_UID, dtype=int)

        if not np.array_equal(np.sort(channel_UID), np.unique(channel_UID)):
            raise ValueError(
                "Duplicate elements found in channelUID list. UID's should be unique!"
            )

        if channel_UID.max() > 1024:
            raise ValueError(
                f"Illegal virtual neuron UID : {channel_UID.max()}! It should be less than 1024"
            )
        if channel_UID.min() < 0:
            raise ValueError(
                f"Illegal UID : {channel_UID.min()}! It should be bigger than 0"
            )

        plt.ylabel("Channels [ChipID, CoreID, NeuronID]")
        plt.yticks(range(len(channel_labels)), [f"s[{key}]" for key in channel_labels])

    return input_sp_ts


def get_tau(
    C: float,
    Itau: float,
    Ut: float = 25e-3,
    kappa_n: float = 0.75,
    kappa_p: float = 0.66,
) -> float:
    """
    get_tau calculates the time constant using the leakage current

    .. math ::
        \\tau = \\dfrac{C U_{T}}{\\kappa I_{\\tau}}

    :param C: capacitor value in farads
    :type C: float
    :param Itau: leakage current in amperes
    :type Itau: float
    :param Ut: termal voltage in volts, defaults to 25e-3
    :type Ut: float, optional
    :param kappa_n: Subthreshold slope factor (n-type transistor), defaults to 0.75
    :type kappa_n: float, optional
    :param kappa_p: Subthreshold slope factor (p-type transistor), defaults to 0.66
    :type kappa_p: float, optional
    :return: time constant in seconds
    :rtype: float
    """
    kappa = (kappa_n + kappa_p) / 2
    tau = (C * Ut) / (kappa * Itau)
    return tau


def Isyn_inf(Ith: float, Itau: float, Iw: float) -> float:
    """
    Isyn_inf calculates the steady state DPI current

    .. math ::
        I_{syn_{\\infty}} = \\dfrac{I_{th}}{I_{\\tau}}I_{w}

    :param Ith: threshold current, a.k.a gain current in amperes
    :type Ith: float
    :param Itau: leakage current in amperes
    :type Itau: float
    :param Iw: weight current in amperes
    :type Iw: float
    :return: steady state current in amperes
    :rtype: float
    """
    Iss = (Ith / Itau) * Iw
    return Iss


def get_param_vector(object_list: ArrayLike, target: str) -> np.ndarray:
    """
    get_param_vector lists the parameters traversing the different object instances of the same class

    :param object_list: list of objects to be traversed
    :type object_list: ArrayLike
    :param target: target parameter to be extracted and listed in the order of object list
    :type target: str
    :return: An array of target parameter values obtained from the objects
    :rtype: np.ndarray
    """
    param_list = [
        obj.__getattribute__(target) if obj is not None else None for obj in object_list
    ]

    return np.array(param_list)


def pulse_width_increment(
    method: str, base_width: float, dt: float
) -> Callable[[int], float]:
    """
    pulse_width_increment defines a method for pulse width increment
    in the case that pulses are merged together as one.
    It can be a logarithic increase which can manage infinite amount of spikes
    or can be a linear increase which is computationally lighter.

    :param method: The increment merhod: "lin" or "log".
    :type method: str
    :param base_width: the unit pulse width to be increased
    :type base_width: float
    :param dt: the simulation timestep
    :type dt: float
    :return: a function to calculate the effective pulse width
    :rtype: Callable[[int], float]
    """
    if method == "log":

        def log_incr(num_spikes: Union[int, np.ndarray]):

            """
            log_incr decreases the increment amount exponentially at each time
            so that the infinite amount of spikes can increase the pulse width
            up to the simulation timestep

            :param num_spikes: number of spikes within one simulation timestep
            :type num_spikes: Union[int, np.ndarray]
            :return: the upgraded pulsewidth
            :rtype: float
            """
            return dt * (1 - np.exp(-num_spikes * base_width / dt))

        return log_incr

    if method == "lin":

        def lin_incr(num_spikes: Union[int, np.ndarray]):
            """
            lin_incr Implements the simplest possible approach. Multiply the
            number of spikes with the unit pulse width

            :param num_spikes: [description]
            :type num_spikes: Union[int, np.ndarray]
            :return: the upgraded pulse width
            :rtype: float
            """
            pulse_width = num_spikes * base_width
            pulse_width = np.clip(pulse_width, 0, dt)
            return pulse_width

        return lin_incr


def pulse_placement(method: str, dt: float) -> Callable[[np.ndarray], float]:
    """
    pulse_placement defines a method to place a pulse inside a larger timebin

    :param method: The method of placement. It can be "middle", "start", "end", or "random"
    :type method: str
    :param dt: the timebin to place the pulse
    :type dt: float
    :return: a function to calculate the time left after the pulse ends
    :rtype: Callable[[np.ndarray],float]
    """
    if method == "middle":

        def _middle(t_pulse: np.ndarray):
            """
            _middle places the pulse right in the middle of the timebin
            .___|-|___.

            :param t_pulse: an array of durations of the pulses
            :type t_pulse: np.ndarray
            :return: the time left after the pulse ends.
            :rtype: float
            """
            t_dis = (dt - t_pulse) / 2.0
            return t_dis

        return _middle

    if method == "start":

        def _start(t_pulse: np.ndarray):
            """
            _start places the pulse at the beginning of the timebin
            .|-|______.

            :param t_pulse: an array of durations of the pulses
            :type t_pulse: np.ndarray
            :return: the time left after the pulse ends.
            :rtype: float
            """
            t_dis = dt - t_pulse
            return t_dis

        return _start

    if method == "end":

        def _end(t_pulse: np.ndarray):
            """
            _end places the pulse at the end of the timebin
            .______|-|.
            Note that it's the most advantageous one becasue
            placing the pulse at the end, one exponential term in the DPI update
            equation can be omitted.

            :param t_pulse: an array of durations of the pulses
            :type t_pulse: np.ndarray
            :return: the time left after the pulse ends.
            :rtype: float
            """
            t_dis = 0
            return t_dis

        return _end

    if method == "random":

        def _random(t_pulse: np.ndarray):
            """
            _random places the pulse to a random place inside the timebin
            .__|-|____.
            ._____|-|_.
            ._|-|_____.

            Note that it's the most expensive one among the other placement methods
            since that there is a random number generation overhead at each time.

            :param t_pulse: an array of durations of the pulses
            :type t_pulse: np.ndarray
            :return: the time left after the pulse ends.
            :rtype: float
            """

            t_pulse_start = (dt - t_pulse) * np.random.random_sample()
            t_dis = dt - t_pulse_start - t_pulse
            return t_dis

        return _random


def dpi_update_func(
    type: str,
    tau: Callable[[], float],
    Iss: Callable[[], float],
    dt: float,
    Io: float,
    t_pulse: Optional[float],
) -> Callable[[float], float]:
    """
    dpi_update_func Returns the DPI update function given the type

    :param type: The update type : 'taylor', 'exp', or 'dpi'
    :type type: str
    :raises TypeError: If the update type given is neither 'taylor' nor 'exp' nor 'dpi'
    :return: a function calculating the value of the synaptic current in the next time step, given the instantaneous synaptic current
    :rtype: Callable[[float], float]
    """
    if type == "taylor":

        def _taylor_update(Isyn, n_spikes):
            Isyn += n_spikes * Iss()
            factor = -dt / tau()
            I_next = Isyn + Isyn * factor
            I_next = jnp.clip(I_next, Io)
            return I_next

        return _taylor_update

    elif type == "exp":

        def _exponential_update(Isyn, n_spikes):
            Isyn += n_spikes * Iss()
            factor = jnp.exp(-dt / tau())
            I_next = Isyn * factor
            I_next = jnp.clip(I_next, Io)
            return I_next

        return _exponential_update

    elif type == "dpi":

        def _dpi_update(Isyn, n_spikes):

            factor = jnp.exp(-dt / tau())
            spikes = n_spikes.astype(bool)

            # CHARGE PHASE
            charge = Iss() * (1.0 - factor) + Isyn * factor
            charge_vector = spikes * charge  # linear

            # DISCHARGE PHASE
            discharge = Isyn * factor
            discharge_vector = (1 - spikes) * discharge

            I_next = charge_vector + discharge_vector
            I_next = jnp.clip(I_next, Io)

            return I_next

        return _dpi_update

    elif type == "dpi_us":  # DPI Undersampled Simulation : only 1 spike allowed in 1ms

        def pulse_width(n_spikes):
            return dt * (1.0 - jnp.exp(-n_spikes * t_pulse / dt))

        def _dpi_us_update(Isyn, n_spikes):

            # t_pulse = pulse_width(n_spikes)
            pw = t_pulse * n_spikes
            full_discharge = jnp.exp(-dt / tau())
            f_charge = jnp.exp(-pw / tau())
            t_dis = (dt - pw) / 2.0
            f_dis = jnp.exp(-t_dis / tau())

            spikes = n_spikes.astype(bool)
            # IF spikes
            # CHARGE PHASE -- UNDERSAMPLED -- dt >> t_pulse
            charge = Iss() * f_dis * (1.0 - f_charge) + Isyn * f_charge * f_dis * f_dis
            charge_vector = spikes * charge

            # IF no spike at all
            # DISCHARGE PHASE
            discharge = Isyn * full_discharge
            discharge_vector = (1 - spikes) * discharge

            I_next = charge_vector + discharge_vector
            I_next = jnp.clip(I_next, Io)

            return I_next

        return _dpi_us_update

    elif type == "dpi_us2":  # DPI Undersampled Simulation : equations simplified

        def pulse_width(n_spikes):
            return dt * (1.0 - jnp.exp(-n_spikes * t_pulse / dt))

        def _random(t_pulse: jnp.ndarray):

            t_pulse_start = (dt - t_pulse) * np.random.random_sample(len(t_pulse))
            t_pulse_end = t_pulse + t_pulse_start
            t_dis = dt - t_pulse_end
            return t_dis

        def _dpi_us2_update(Isyn, n_spikes):

            f_dt = jnp.exp(-dt / tau())

            # t_pulse = pulse_width(n_spikes)
            pw = t_pulse * n_spikes

            # t_dis = (dt - t_pulse) / 2.0
            t_dis = _random(pw)
            f_charge = jnp.exp(-t_dis / tau()) - jnp.exp(-(t_dis + pw) / tau())

            # DISCHARGE IN ANY CASE
            Isyn *= f_dt

            # CHARGE PHASE -- UNDERSAMPLED -- dt >> t_pulse
            spikes = n_spikes.astype(bool)
            Isyn += Iss() * (f_charge * spikes)

            Isyn = jnp.clip(Isyn, Io)

            return Isyn

        return _dpi_us2_update

    elif (
        type == "dpi_us3"
    ):  # DPI Undersampled Simulation : more simplification : t_dis = 0, pulse is at the end

        def pulse_width(n_spikes):
            return dt * (1.0 - jnp.exp(-n_spikes * t_pulse / dt))

        def _dpi_us3_update(Isyn, n_spikes):
            f_dt = jnp.exp(-dt / tau())
            # t_pulse = pulse_width(n_spikes)
            pw = t_pulse * n_spikes
            f_charge = 1 - jnp.exp(-pw / tau())

            # DISCHARGE IN ANY CASE
            Isyn *= f_dt

            # CHARGE PHASE -- UNDERSAMPLED -- dt >> t_pulse
            # f_charge array = 0 where there is no spike
            Isyn += Iss() * f_charge

            Isyn = jnp.clip(Isyn, Io)

            return Isyn

        return _dpi_us3_update
    else:
        raise TypeError(
            f"{type} Update type undefined. Try one of 'taylor', 'exp', 'dpi'"
        )


def set_param(
    shape: tuple, family: str, init_func: Callable, object: str
) -> JP_ndarray:
    """
    set_param is a utility function helps making a neat selection of state, parameter or simulation parameter
    Seee Also:
        See :py:class:`.Parameter` for representing the configuration of a module, :py:class:`.State` for representing the transient internal state of a neuron or module, and :py:class:`.SimulationParameter` for representing simulation- or solver-specific parameters that are not important for network configuration.

    :param shape: Specifying the permitted shape of the attribute.
    :type shape: tuple
    :param family: An arbitrary string to specify the "family" of this attribute. ``'weights'``, ``'taus'``, ``'biases'`` are popular choices.
    :type family: str
    :param init_func: A function that initializes the attribute of interest
    :type init_func: Callable
    :param object: the type of the object to be constructed. It can be "state", "parameter" or "simulation"
    :type object: str
    :raises ValueError: When object type is not one of "state", "parameter" or "simulation"
    :return: constructed parameter or the state variable
    :rtype: JP_ndarray
    """
    if object.upper() == "STATE":
        Iparam: JP_ndarray = State(shape=shape, family=family, init_func=init_func)
        return Iparam
    elif object.upper() == "PARAMETER":
        Iparam: JP_ndarray = Parameter(shape=shape, family=family, init_func=init_func)
        return Iparam
    elif object.upper() == "SIMULATION":
        Iparam: JP_ndarray = SimulationParameter(
            shape=shape, family=family, init_func=init_func
        )
        return Iparam
    else:
        raise ValueError(
            f"object type: {object} can be 'state', 'parameter', or 'simulation'"
        )


def bias_current_table(
    mod: DynapSE1Jax,
    chipID: np.uint8,
    coreID: np.uint8,
    reciprocals: List[Tuple[str]],
    default_mod: Optional[DynapSE1Jax] = None,
) -> Dict[str, List[Union[str, Tuple[int], float]]]:
    """
    bias_current_table creates a table in the form of a dictionary. It includes the bias parameters, coarse and fine values,
    corresponding simulation currents, ampere values and nominal simulation values for comparison purposes.
    The dicionary can easily be converted to a pandas dataframe if desired.

    :param mod: the module to be investigated
    :type mod: DynapSE1Jax
    :param chipID: Unique chip ID to of the simulation module to examine
    :type chipID: np.uint8
    :param coreID: Non-unique core ID to of the simulation module to examine
    :type coreID: np.uint8
    :param reciprocals: a mapping from bias parameters to simualtion currents. Looks like [("PS_WEIGHT_INH_S_N", "Iw_gaba_b")]
    :type reciprocals: List[Tuple[str]]
    :param default_mod: a default simulation module to extract nominal simulation values, defaults to None
    :type default_mod: Optional[DynapSE1Jax], optional
    :return: a dictionary encapsulating all the bias current simulation current information extracted depending on reciprocal list

    e.g. (The dictionary converted to a dataframe `pd.DataFrame(data)`)
                        Bias Coarse,Fine    Current  Amperes  Nominal(A)
        0  PS_WEIGHT_INH_S_N      (0, 0)  Iw_gaba_b 0.00e+00    1.00e-07
        1  PS_WEIGHT_INH_F_N    (7, 255)  Iw_gaba_a 2.40e-05    1.00e-07
        2  PS_WEIGHT_EXC_S_N     (6, 82)    Iw_nmda 1.03e-06    1.00e-07
        3  PS_WEIGHT_EXC_F_N    (7, 219)    Iw_ampa 2.06e-05    1.00e-07
        4           IF_AHW_P      (0, 0)     Iw_ahp 0.00e+00    1.00e-07
        5            IF_DC_P    (1, 254)        Idc 1.05e-10    5.00e-13
        6          IF_NMDA_N      (0, 0)    If_nmda 0.00e+00    5.00e-13

    :rtype: Dict[str, List[Union[str, Tuple[int], float]]]
    """
    if default_mod is None:
        default_mod = DynapSE1Jax((1, 1))

    # Construct the dictionary
    keys = ["Bias", "Coarse,Fine", "Current", "Amperes", "Nominal(A)"]
    data = dict(zip(keys, [list() for i in keys]))
    data["Bias"], data["Current"] = map(list, zip(*reciprocals))

    # Fill bias coarse and fine values
    for bias in data["Bias"]:
        param = getattr(mod, bias)(chipID, coreID)
        data["Coarse,Fine"].append((param.coarse_value, param.fine_value))

    # Fill current values in amperes and also the nominal values
    for sim_current in data["Current"]:
        data["Amperes"].append(mod.get_bias(chipID, coreID, sim_current))
        data["Nominal(A)"].append(default_mod.get_bias(0, 0, sim_current))

    return data


def high_level_parameter_table(
    mod: DynapSE1Jax,
    chipID: np.uint8,
    coreID: np.uint8,
    param_list: List[str],
    default_mod: Optional[DynapSE1Jax] = None,
) -> Dict[str, List[Union[str, Tuple[int], float]]]:
    """
    high_level_parameter_table creates a table in the form of a dictionary. It includes high level parameters, their values
    and also the nominal simulation values for comparison purposes.
    The dicionary can easily be converted to a pandas dataframe if desired.

    :param mod: the module to be investigated
    :type mod: DynapSE1Jax
    :param chipID: Unique chip ID to of the simulation module to examine
    :type chipID: np.uint8
    :param coreID: Non-unique core ID to of the simulation module to examine
    :type coreID: np.uint8
    :param param_list: a list of high level parameters to be obtained from the module and stored in a table-compatible dictionary. e.g ["tau_mem", "tau_ampa"]
    :type param_list: List[str]
    :param default_mod: a default simulation module to extract nominal simulation values, defaults to None
    :type default_mod: Optional[DynapSE1Jax], optional
    :return: a dictionary encapsulating all the high level parameter information extracted depending on param list

    e.g. (The dictionary converted to a dataframe `pd.DataFrame(data)`)
            Parameter      Value  Nominal
        0     tau_mem   1.26e-06 2.00e-02
        1  tau_gaba_b   4.05e-03 1.00e-01
        2  tau_gaba_a   3.62e-08 1.00e-02
        3    tau_nmda   1.27e-02 1.00e-01
        4    tau_ampa   5.40e-03 1.00e-02
        5     tau_ahp   9.04e-05 5.00e-02
        6       t_ref   3.55e-06 1.00e-02
        7     t_pulse   2.96e-06 1.00e-05

    :rtype: Dict[str, List[Union[str, Tuple[int], float]]]
    """

    if default_mod is None:
        default_mod = DynapSE1Jax((1, 1))

    # Construct the dictionary
    keys = ["Parameter", "Value", "Nominal"]
    data = dict(zip(keys, [list() for i in keys]))

    data["Parameter"] = param_list

    for param in data["Parameter"]:
        data["Value"].append(mod.get_bias(chipID, coreID, param))
        data["Nominal"].append(default_mod.get_bias(0, 0, param))

    return data


def merge_biases_high_level(
    mod: DynapSE1Jax,
    chipID: np.uint8,
    coreID: np.uint8,
    reciprocals: List[Tuple[str]],
    default_mod: Optional[DynapSE1Jax] = None,
) -> Dict[str, List[Union[str, Tuple[int], float]]]:
    """
    merge_biases_high_level merge the bias current&simulation currents table data and high level parameter data.
    It can be considered as a wrapper for using  `bias_current_table()` and `high_level_parameter_table()` together.
    The tuples in the reciprocal list should be like `("IF_TAU1_N", "Itau_mem", "tau_mem")`. First the bias param,
    then corresponding simulation current, then the related high-level parameter.

    :param mod: the module to be investigated
    :type mod: DynapSE1Jax
    :param chipID: Unique chip ID to of the simulation module to examine
    :type chipID: np.uint8
    :param coreID: Non-unique core ID to of the simulation module to examine
    :type coreID: np.uint8
    :param reciprocals: a mapping from bias parameters to simualtion currents. Looks like [("PS_WEIGHT_INH_S_N", "Iw_gaba_b")]
    :type reciprocals: List[Tuple[str]]
    :param default_mod: a default simulation module to extract nominal simulation values, defaults to None
    :type default_mod: Optional[DynapSE1Jax], optional
    :return: a dictionary encapsulating all the bias current simulation current information, and their related higher level parameter information extracted depending on reciprocal list
    :rtype: Dict[str, List[Union[str, Tuple[int], float]]]
    """

    bias_current, parameter = map(
        list, zip(*list(map(lambda t: ((t[0], t[1]), t[2]), reciprocals)))
    )

    # Obtain bias + high-level parameters table
    data = bias_current_table(mod, chipID, coreID, bias_current, default_mod)
    taus = high_level_parameter_table(mod, chipID, coreID, parameter, default_mod)

    data.update(taus)

    return data


def time_const_table(
    mod: DynapSE1Jax,
    chipID: np.uint8,
    coreID: np.uint8,
    default_mod: Optional[DynapSE1Jax] = None,
    float_format: str = "{:,.2e}".format,
) -> pd.DataFrame:
    """
    time_const_table creates a pandas dataframe to investigate the consequences of
    the bias currents which have a corresponding time constant used in the simulator.
    ['IF_TAU1_N', 'NPDPII_TAU_S_P', 'NPDPII_TAU_F_P', 'NPDPIE_TAU_S_P', 'NPDPIE_TAU_F_P', 'IF_AHTAU_N', 'IF_RFR_N', 'PULSE_PWLK_P']

    :param mod: the module to be investigated
    :type mod: DynapSE1Jax
    :param chipID: Unique chip ID to of the simulation module to examine
    :type chipID: np.uint8
    :param coreID: Non-unique core ID to of the simulation module to examine
    :type coreID: np.uint8
    :param default_mod: a default simulation module to extract nominal simulation values, defaults to None
    :type default_mod: Optional[DynapSE1Jax], optional
    :param float_format: the float printing format used printing the dataframe, defaults to "{:,.2e}".format
    :type float_format: str, optional
    :return: a table for examining the bias currents which sets some time constants on the device.

    e.g.
                    Bias Coarse,Fine      Current  Amperes  Nominal(A)   Parameter     Value  Nominal
        0       IF_TAU1_N     (5, 90)     Itau_mem 1.41e-07    8.87e-12     tau_mem  1.26e-06 2.00e-02
        1  NPDPII_TAU_S_P     (2, 68)  Itau_gaba_b 2.19e-10    8.87e-12  tau_gaba_b  4.05e-03 1.00e-01
        2  NPDPII_TAU_F_P    (7, 255)  Itau_gaba_a 2.40e-05    8.69e-11  tau_gaba_a  3.62e-08 1.00e-02
        3  NPDPIE_TAU_S_P    (1, 169)    Itau_nmda 6.96e-11    8.87e-12    tau_nmda  1.27e-02 1.00e-01
        4  NPDPIE_TAU_F_P     (2, 50)    Itau_ampa 1.61e-10    8.69e-11    tau_ampa  5.40e-03 1.00e-02
        5      IF_AHTAU_N     (4, 80)     Itau_ahp 1.57e-08    2.84e-11     tau_ahp  9.04e-05 5.00e-02
        6        IF_RFR_N    (3, 196)         Iref 5.00e-09    1.77e-12       t_ref  3.55e-06 1.00e-02
        7    PULSE_PWLK_P    (3, 235)       Ipulse 5.99e-09    1.77e-09     t_pulse  2.96e-06 1.00e-05

    :rtype: pd.DataFrame
    """

    if default_mod is None:
        default_mod = DynapSE1Jax((1, 1))

    # Hard-coded reciprocal list
    reciprocals = [
        ("IF_TAU1_N", "Itau_mem", "tau_mem"),
        ("NPDPII_TAU_S_P", "Itau_gaba_b", "tau_gaba_b"),
        ("NPDPII_TAU_F_P", "Itau_gaba_a", "tau_gaba_a"),
        ("NPDPIE_TAU_S_P", "Itau_nmda", "tau_nmda"),
        ("NPDPIE_TAU_F_P", "Itau_ampa", "tau_ampa"),
        ("IF_AHTAU_N", "Itau_ahp", "tau_ahp"),
        ("IF_RFR_N", "Iref", "t_ref"),
        ("PULSE_PWLK_P", "Ipulse", "t_pulse"),
    ]

    data = merge_biases_high_level(mod, chipID, coreID, reciprocals, default_mod)

    pd.options.display.float_format = float_format
    return pd.DataFrame(data)


def gain_table(
    mod: DynapSE1Jax,
    chipID: np.uint8,
    coreID: np.uint8,
    default_mod: Optional[DynapSE1Jax] = None,
    float_format: str = "{:,.2e}".format,
) -> pd.DataFrame:
    """
    gain_table creates a pandas dataframe to investigate the consequences of
    the bias currents which have a corresponding gain factor used in the simulator.
    ['IF_THR_N', 'NPDPII_THR_S_P', 'NPDPII_THR_F_P', 'NPDPIE_THR_S_P', 'NPDPIE_THR_F_P', 'IF_AHTHR_N']

    :param mod: the module to be investigated
    :type mod: DynapSE1Jax
    :param chipID: Unique chip ID to of the simulation module to examine
    :type chipID: np.uint8
    :param coreID: Non-unique core ID to of the simulation module to examine
    :type coreID: np.uint8
    :param default_mod: a default simulation module to extract nominal simulation values, defaults to None
    :type default_mod: Optional[DynapSE1Jax], optional
    :param float_format: the float printing format used printing the dataframe, defaults to "{:,.2e}".format
    :type float_format: str, optional
    :return: a table for examining the bias currents which sets the gain factors of the DPI circuits on the device.

    e.g.
                     Bias Coarse,Fine     Current  Amperes  Nominal(A)    Parameter     Value  Nominal   f_gain  Nominal(1)
        0        IF_THR_N    (2, 254)     Ith_mem 8.17e-10    3.55e-11     Itau_mem  1.04e-08 8.87e-12 7.86e-02    4.00e+00
        1  NPDPII_THR_S_P    (7, 255)  Ith_gaba_b 2.40e-05    3.55e-11  Itau_gaba_b  2.40e-05 8.87e-12 1.00e+00    4.00e+00
        2  NPDPII_THR_F_P     (2, 44)  Ith_gaba_a 1.41e-10    3.48e-10  Itau_gaba_a  7.00e-11 8.69e-11 2.02e+00    4.00e+00
        3  NPDPIE_THR_S_P     (3, 38)    Ith_nmda 9.69e-10    3.55e-11    Itau_nmda  4.76e-10 8.87e-12 2.04e+00    4.00e+00
        4  NPDPIE_THR_F_P     (5, 46)    Ith_ampa 7.22e-08    3.48e-10    Itau_ampa  3.63e-08 8.69e-11 1.99e+00    4.00e+00
        5      IF_AHTHR_N    (4, 161)     Ith_ahp 3.16e-08    1.13e-10     Itau_ahp  1.57e-08 2.84e-11 2.01e+00    4.00e+00

    :rtype: pd.DataFrame
    """

    if default_mod is None:
        default_mod = DynapSE1Jax((1, 1))

    # Hard-coded reciprocal list
    reciprocals = [
        ("IF_THR_N", "Ith_mem", "Itau_mem"),
        ("NPDPII_THR_S_P", "Ith_gaba_b", "Itau_gaba_b"),
        ("NPDPII_THR_F_P", "Ith_gaba_a", "Itau_gaba_a"),
        ("NPDPIE_THR_S_P", "Ith_nmda", "Itau_nmda"),
        ("NPDPIE_THR_F_P", "Ith_ampa", "Itau_ampa"),
        ("IF_AHTHR_N", "Ith_ahp", "Itau_ahp"),
    ]

    data = merge_biases_high_level(mod, chipID, coreID, reciprocals, default_mod)

    # Add gain rows
    keys = ["f_gain", "Nominal(1)"]
    data.update(dict(zip(keys, [list() for i in keys])))

    # Iterate over existing data and calculate
    for Igain, Ileak, Igain_nom, Ileak_nom in zip(
        data["Amperes"], data["Value"], data["Nominal(A)"], data["Nominal"]
    ):
        data["f_gain"].append(Igain / Ileak)
        data["Nominal(1)"].append(Igain_nom / Ileak_nom)

    pd.options.display.float_format = float_format
    return pd.DataFrame(data)


def synapse_table(
    mod: DynapSE1Jax,
    chipID: np.uint8,
    coreID: np.uint8,
    default_mod: Optional[DynapSE1Jax] = None,
    float_format: str = "{:,.2e}".format,
) -> pd.DataFrame:
    """
    synapse_table creates a pandas dataframe to investigate the consequences of the bias currents which have a
    corresponding synapse weight current or a current affecting the synapse weight indirectly and used in the simulator.
    ['PS_WEIGHT_INH_S_N', 'PS_WEIGHT_INH_F_N', 'PS_WEIGHT_EXC_S_N', 'PS_WEIGHT_EXC_F_N', 'IF_AHW_P', 'IF_DC_P', 'IF_NMDA_N']

    :param mod: the module to be investigated
    :type mod: DynapSE1Jax
    :param chipID: Unique chip ID to of the simulation module to examine
    :type chipID: np.uint8
    :param coreID: Non-unique core ID to of the simulation module to examine
    :type coreID: np.uint8
    :param default_mod: a default simulation module to extract nominal simulation values, defaults to None
    :type default_mod: Optional[DynapSE1Jax], optional
    :param float_format: the float printing format used printing the dataframe, defaults to "{:,.2e}".format
    :type float_format: str, optional
    :return: a table for examining the bias currents which affects the synaptic weights of the DPI circuits on the device.

    e.g.

                        Bias Coarse,Fine    Current  Amperes  Nominal(A)
        0  PS_WEIGHT_INH_S_N      (0, 0)  Iw_gaba_b 0.00e+00    1.00e-07
        1  PS_WEIGHT_INH_F_N    (7, 255)  Iw_gaba_a 2.40e-05    1.00e-07
        2  PS_WEIGHT_EXC_S_N     (6, 82)    Iw_nmda 1.03e-06    1.00e-07
        3  PS_WEIGHT_EXC_F_N    (7, 219)    Iw_ampa 2.06e-05    1.00e-07
        4           IF_AHW_P      (0, 0)     Iw_ahp 0.00e+00    1.00e-07
        5            IF_DC_P    (1, 254)        Idc 1.05e-10    5.00e-13
        6          IF_NMDA_N      (0, 0)    If_nmda 0.00e+00    5.00e-13


    :rtype: pd.DataFrame
    """

    if default_mod is None:
        default_mod = DynapSE1Jax((1, 1))

    # Hard-coded reciprocal list
    reciprocals = [
        ("PS_WEIGHT_INH_S_N", "Iw_gaba_b"),
        ("PS_WEIGHT_INH_F_N", "Iw_gaba_a"),
        ("PS_WEIGHT_EXC_S_N", "Iw_nmda"),
        ("PS_WEIGHT_EXC_F_N", "Iw_ampa"),
        ("IF_AHW_P", "Iw_ahp"),
        ("IF_DC_P", "Idc"),
        ("IF_NMDA_N", "If_nmda"),
    ]

    data = bias_current_table(mod, chipID, coreID, reciprocals, default_mod)

    pd.options.display.float_format = float_format
    return pd.DataFrame(data)


def bias_table(
    mod: DynapSE1Jax,
    chipID: np.uint8,
    coreID: np.uint8,
    default_mod: Optional[DynapSE1Jax] = None,
    float_format: str = "{:,.2e}".format,
) -> pd.DataFrame:
    """
    bias_table merges `time_const_table`, `gain_table`, and `synapse_table` together in one table by
    providing proper keys for each one and represent all the simulated biases within one core in categories.

    :param mod: the module to be investigated
    :type mod: DynapSE1Jax
    :param chipID: Unique chip ID to of the simulation module to examine
    :type chipID: np.uint8
    :param coreID: Non-unique core ID to of the simulation module to examine
    :type coreID: np.uint8
    :param default_mod: a default simulation module to extract nominal simulation values, defaults to None
    :type default_mod: Optional[DynapSE1Jax], optional
    :param float_format: the float printing format used printing the dataframe, defaults to "{:,.2e}".format
    :type float_format: str, optional
    :return: a table for examining all the bias currents configuring the network on the device.
    :rtype: pd.DataFrame
    """

    if default_mod is None:
        default_mod = DynapSE1Jax((1, 1))

    # Generate Tables
    syn_tab = synapse_table(mod, chipID, coreID, default_mod, float_format)
    time_tab = time_const_table(mod, chipID, coreID, default_mod, float_format)
    gain_tab = gain_table(mod, chipID, coreID, default_mod, float_format)

    bias_tab = pd.concat(
        [time_tab, gain_tab, syn_tab], keys=["Time Const.", "Gain", "Synapses"]
    )
    return bias_tab


def device_vs_simulation(
    mod: DynapSE1Jax,
    default_mod: Optional[DynapSE1Jax] = None,
    float_format: str = "{:,.2e}".format,
) -> pd.DataFrame:
    """
    device_vs_simulation merges `bias_table`s for each active core together in one table by
    providing proper keys for each one and represent all the simulated biases in categories.

    :param mod: the module to be investigated
    :type mod: DynapSE1Jax
    :param default_mod: a default simulation module to extract nominal simulation values, defaults to None
    :type default_mod: Optional[DynapSE1Jax], optional
    :param float_format: the float printing format used printing the dataframe, defaults to "{:,.2e}".format
    :type float_format: str, optional
    :return: [description]
    :rtype: pd.DataFrame
    """

    if default_mod is None:
        default_mod = DynapSE1Jax((1, 1))

    tables = []
    keys = []

    # Iterate through the active cores
    for chipID, coreID in list(mod.core_dict.keys()):
        tables.append(bias_table(mod, chipID, coreID, default_mod, float_format))
        keys.append(f"C{chipID}c{coreID}")

    comp_tab = pd.concat(tables, keys=keys)

    return comp_tab
