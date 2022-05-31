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
from rockpool.devices.dynapse.definitions import ArrayLike

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


def poisson_spike_train(
    n_channels: int,
    duration: float,
    rate: float,
    dt: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    random_spike_train generates a Poisson frozen random spike train

    :param n_channels: number of channels
    :type n_channels: float
    :param duration: simulation duration in seconds
    :type duration: float
    :param rate: expected mean spiking rate in Hertz(1/s)
    :type rate: float
    :param dt: time step length
    :type dt: float, optional
    :param seed: the random number seed
    :type seed: int, optional
    :raises ValueError: no spike generated due to low firing rate or very short simulation time]
    :return: randomly generated discrete spike train
    :rtype: np.ndarray
    """
    np.random.seed(seed)
    steps = int(np.round(duration / dt))
    raster = np.random.poisson(rate * dt, (steps, n_channels))

    # Check if raster has at least one spike
    if not any(raster.flatten()):
        raise ValueError(
            "No spike generated at all due to low firing rate or short simulation time duration!"
        )

    spike_tensor = np.array(raster, dtype=float)
    return spike_tensor
