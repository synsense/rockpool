"""
Helpful functions for performing analysis on spiking outputs
"""


import numpy as np
from typing import List

from .timeseries import TSEvent


def lv(tsevents: TSEvent) -> np.ndarray:
    """
    lv - Calculates the lv measure for each channel

    :return: np.ndarray with lv measure for each channel
    """
    lv_list = np.zeros(tsevents.num_channels)
    for channel_id in range(tsevents.num_channels):
        times: np.ndarray = tsevents.times[np.where(tsevents.channels == channel_id)[0]]
        time_intervals: np.ndarray = np.diff(times)
        lo_var = 3.0 * np.mean(
            np.power(
                np.diff(time_intervals) / (time_intervals[:-1] + time_intervals[1:]), 2
            )
        )
        lv_list[channel_id] = lo_var

    return lv_list


def fano_factor(tsevents: TSEvent, dt: float = 0.001) -> float:
    """
    FanoFactor() - put as input a spike detector nest object and return mean FanoFactor of the network

    :param dt: float raster timestep in sec
    :return: float FanoFactor
    """
    raster = tsevents.raster(dt, add_events=True).T
    hist = raster.sum(axis=0)
    return np.var(hist) / np.mean(hist)
