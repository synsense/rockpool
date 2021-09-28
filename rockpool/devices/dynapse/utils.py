"""
Utility functions for Dynap-SE1/SE2 simulator

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
23/07/2021
"""

import numpy as np
from rockpool import TSEvent, TSContinuous

from typing import (
    Callable,
    Optional,
    Union,
    Tuple,
    Generator,
    List,
    Dict,
)

import jax.numpy as jnp

from rockpool.parameters import Parameter, State, SimulationParameter
from rockpool.typehints import JP_ndarray, P_float
from rockpool.devices.dynapse.biasgen import BiasGen

_SAMNA_AVAILABLE = True

try:
    from samna.dynapse1 import Dynapse1Parameter
except ModuleNotFoundError as e:
    print(
        e,
        "\nDynapSE1NeuronSynapseJax module can only be used for simulation purposes."
        "Deployment utilities depends on samna!",
    )
    _SAMNA_AVAILABLE = False

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
    channel_UID: Optional[ArrayLike] = None,
    return_channel_UID: bool = False,
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
    :param channel_UID: a list of universal neuron IDs to assign channels, defaults to None
    :type channel_UID: Optional[ArrayLike], optional
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
    spiking_prob = rate * dt
    input_sp_raster = np.random.rand(steps, n_channels) < spiking_prob
    if not any(input_sp_raster.flatten()):
        raise ValueError(
            "No spike generated at all due to low firing rate or short simulation time duration!"
        )
    input_sp_ts = TSEvent.from_raster(input_sp_raster, name=name, periodic=True, dt=dt)

    if channel_UID is not None:
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

        channel_UID = channel_UID.astype(np.uint16)

        return_channel_UID = True

    elif return_channel_UID:
        channel_UID = np.array(
            list(range(1, input_sp_ts.num_channels + 1)), dtype=np.uint16
        )

        channel_UID_dict = dict(zip(range(input_sp_ts.num_channels), channel_UID))

    if return_channel_UID:
        return input_sp_ts, channel_UID_dict
    else:
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


def get_Dynapse1Parameter(bias: float, name: str) -> Dynapse1Parameter:
    """
    get_Dynapse1Parameter constructs a samna DynapSE1Parameter object given the bias current desired

    :param bias: bias current desired. It will be expressed by a coarse fine tuple which will generate the closest possible bias current.
    :type bias: float
    :param name: the name of the bias parameter
    :type name: str
    :return: samna DynapSE1Parameter object
    :rtype: Dynapse1Parameter
    """
    coarse, fine = BiasGen.bias_to_coarse_fine(bias)
    param = Dynapse1Parameter(name, coarse, fine)
    return param
