# -----------------------------------------------------------
# This module implements the ADC as a state machine.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
#
# last update: 30.03.2023
# -----------------------------------------------------------

import numpy as np
from typing import Any


# ===========================================================================
# *    some constants defined according to Xylo-A3 specficiations
# ===========================================================================
from agc.xylo_a3_agc_specs import XYLO_MAX_AMP, AUDIO_SAMPLING_RATE


class ADC:
    def __init__(
        self,
        num_bits: int = 10,
        max_audio_amplitude: float = XYLO_MAX_AMP,
        fs: float = AUDIO_SAMPLING_RATE,
    ):
        """this module implements ADC as a state machine.

        Args:
            fs (float): sample rate of the incoming signal.
        """
        self.num_bits = num_bits
        self.max_audio_amplitude = max_audio_amplitude
        self.fs = fs

        self.reset()

    def reset(self):
        self.time_stamp = 0
        self.sample = 0
        self.stable_sig_in = 0

        # number of samples receied
        self.num_processed_samples = 0

        # reset the state as well
        self.state = {}

    def evolve(self, sig_in: float, time_in: float, record: bool = False):
        """this function processes the input timed-sample and updates the state of ADC.

        Args:
            sig_in (float): input signal sample.
            time_in (float): time stamp of the signal sample.
            record (bool, optional): record simulation state. Defaults to False.
        """
        # NOTE: if PGA is not settled after gain change, its output signal will be unstable.
        # In such a case, we model the output of PGA by None.
        # When this happens, we assume that there is a buffer that keeps the stable value of ADC and avoid accepting new samples from ADC until
        # PGA returns to the stable mode. During this unstable period, the buffer keeps sending the last stable sample it received from ADC.

        if sig_in is None:
            # just repeat the previous sample since the input received from the amplifier is invalid
            sig_in = self.stable_sig_in
        else:
            # if stable record the sample for the next time instants
            self.stable_sig_in = sig_in

        # check the start of the simulation and set the gain values in PGA
        if self.num_processed_samples == 0:
            if record:
                self.state = {
                    "num_processed_samples": [],
                    "time_in": [],
                    "adc_in": [],
                    "adc_out": [],
                }
            else:
                self.state = {}

        # increase the number of processed samples
        self.num_processed_samples += 1

        # record the state if needed
        if record:
            self.state["num_processed_samples"].append(self.num_processed_samples)
            self.state["time_in"].append(time_in)
            self.state["adc_in"].append(sig_in)

        if time_in >= self.time_stamp / self.fs:
            # * it is time to get the quantized version of the sample
            EPS = 0.00001
            sig_in_norm = sig_in / (XYLO_MAX_AMP * (1 + EPS))

            # add a one unit of clock delay to the returned sample
            sample_return, self.sample = self.sample, int(
                np.fix(2 ** (self.num_bits - 1) * sig_in_norm)
            )

            self.time_stamp += 1

            # record the state: returned sample
            if record:
                self.state["adc_out"].append(sample_return)

            return sample_return

        else:
            # * otherwise: record the state and return previously stored sample
            if record:
                self.state["adc_out"].append(self.sample)

            # just return the previously registered sample
            return self.sample

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        this is the same ads evolve function.
        """
        return self.evolve(*args, **kwargs)

    def __repr__(self) -> str:
        # string representation of the ADC module
        string = (
            "+" * 100
            + "\n"
            + "ADC module:\n"
            + f"clock rate: {self.fs}\n"
            + f"maximum amplitude: {self.max_audio_amplitude}\n"
            + f"number of bits used for signed quantization: {self.num_bits}\n"
        )

        return string
