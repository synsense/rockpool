"""
Utility functions for Dynap-SE1/SE2 simulator

renamed: utils.py -> spike_input.py @ 220114

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
23/07/2021
"""
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from rockpool.timeseries import TSEvent, TSContinuous

from rockpool.devices.dynapse.infrastructure.router import Router
from rockpool.devices.dynapse.base import ArrayLike

import matplotlib.pyplot as plt


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
    input_sp_raster = np.random.poisson(rate * dt, (steps, n_channels))
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
