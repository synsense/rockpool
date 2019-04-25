import random
import numpy as np


class SignalGen:
    """
    SignalGen - Generate signal consisting of periods of 3 (normal) or 2 (abnormal) pulses.
    """

    def __init__(self, width_pulse: int, period: int, p_abnormal: float):
        # - Store data
        if width_pulse < period // 4:
            self.width_pulse = int(width_pulse)
            self.period = int(period)
        else:
            raise ValueError("period must be at least 4 * width_pulse")
        if 0 <= p_abnormal <= 1:
            self.p_abnormal = p_abnormal
        else:
            raise ValueError("p_abnormal must be between 0 and 1.")

    def __call__(self, num_segments):
        # - Generate normal and abnormal segments
        normal_segment = np.zeros(self.period)
        abnormal_segment = np.zeros(self.period)
        # Which time points correspond to pulses 1 and 3
        odd_pulses = np.arange(self.period) % (self.period / 2) < self.width_pulse
        # Which time points correspond to pulse 2
        pulse_2_idx = (
            np.where(np.arange(self.period) < self.width_pulse)[0] + self.period // 4
        )
        normal_segment[odd_pulses] = 1
        normal_segment[pulse_2_idx] = 1
        abnormal_segment[odd_pulses] = 1

        # - Generate signal by putting normal and abnormal segments randomly together
        segments_isnormal = np.random.choice(
            (True, False), size=num_segments, p=(1 - self.p_abnormal, self.p_abnormal)
        )
        segment_list = [
            normal_segment if normal else abnormal_segment
            for normal in segments_isnormal
        ]
        target_list = [
            np.zeros(self.period) if normal else np.ones(self.period)
            for normal in segments_isnormal
        ]
        signal = np.array(segment_list).flatten()
        target = np.array(target_list).flatten()
        return signal, target


def signal_to_spikes(signal, spikes_per_edge: int, ts_per_spike: int):
    """
    signal_to_spikes - Convert input signal (pulses) to spikes, usind delta modulation.
                       Effectively, when there is a rising edge a bunch of spikes is
                       generated on channel 0, on a falling edge on channel 1.
    :param signal:  The signal to be converted. Must be of the form as produced when
                    calling SignalGen class instance.
    :param spikes_per_edge:  Spikes to be generated per edge of a pulse
    :ts_per_spike:  (Minimum) number of timesteps between individual spikes
    """
    # - Determine positions of rising and falling edges
    rising_idx = np.where(np.diff(signal) > 0)[0] + 1
    falling_idx = np.where(np.diff(signal) < 0)[0] + 1
    # - Spike times if edge were at 0:
    spiketimes_single = np.arange(spikes_per_edge) * ts_per_spike
    # - Spike times for channel 0 (rising) and for channel 1 (falling)
    times_0 = [spiketimes_single + edgetime for edgetime in rising_idx]
    times_1 = [spiketimes_single + edgetime for edgetime in falling_idx]
    times_01 = np.array(times_0 + times_1).flatten()
    # - Channels
    channels_01 = np.zeros(times_01.size)
    channels_01[-len(times_1) * spikes_per_edge :] = 1
    # - Sort by time
    sort_idx = np.argsort(times_01)
    times = times_01[sort_idx]
    channels = channels_01[sort_idx]

    return times, channels
