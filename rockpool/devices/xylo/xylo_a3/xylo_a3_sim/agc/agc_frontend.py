# -----------------------------------------------------------
# This module implements the quantization module consisting of amplifier, AGC, and ADC.
# The output is the quantized signal coming out of ADC.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 14.06.2023
# -----------------------------------------------------------
import numpy as np

from rockpool.devices.xylo.xylo_a3.xylo_a3_sim.agc.amplifier import Amplifier
from rockpool.devices.xylo.xylo_a3.xylo_a3_sim.agc.adc import ADC
from rockpool.devices.xylo.xylo_a3.xylo_a3_sim.agc.envelope_controller import EnvelopeController
from rockpool.devices.xylo.xylo_a3.xylo_a3_sim.agc.gain_smoother import GainSmootherFPGA

from rockpool.devices.xylo.xylo_a3.xylo_a3_sim.agc.xylo_a3_agc_specs import AUDIO_SAMPLING_RATE

from typing import Any
from tqdm import tqdm
from numpy.linalg import norm


class AGC_ADC:
    def __init__(
        self,
        amplifier: Amplifier,
        adc: ADC,
        envelope_controller: EnvelopeController,
        gain_smoother: GainSmootherFPGA = None,
    ):
        """this module uses the modules designed for AGC and simulates them.

        Args:
            amplifier (Amplifier): programmable gain amplifier (PGA) used for signal amplification (after the first fixed-gain amplifier).
            adc (ADC): ADC module used for quantizing the PGA output.
            envelope_controller (EnvelopeController): module used for detecting the envelope of the signal and adjusting it based on the gain-commands
            it sends to PGA to adjust the gain.
            gain_smoother (GainSmootherFPGA, optional): module used to smooth out the gain to avoid gain jumps. Defaults to None.

        """
        # ================================================
        # *   some sanity check on the modules
        # ================================================
        if amplifier.max_audio_amplitude != adc.max_audio_amplitude:
            raise ValueError(
                "amplifier and ADC have different maximum amplitudes! They should be the same!"
            )

        EPS = 0.00001
        if gain_smoother is not None:
            if (
                norm(amplifier.pga_gain_vec - gain_smoother.pga_gain_vec)
                / np.min(
                    [norm(amplifier.pga_gain_vec), norm(gain_smoother.pga_gain_vec)]
                )
                > EPS
            ):
                raise ValueError(
                    "amplifier, envelope controller, and gain smoother modules should use the same set of gain vectors!"
                )

        self.amplifier = amplifier
        self.adc = adc
        self.envelope_controller = envelope_controller
        self.gain_smoother = gain_smoother

        # =============================================================================
        # * check if the simulation clocks of modules match
        #   NOTE: this is needed since the amplifier typically needs to be simulated
        #   with a higher sampling rate.
        # =============================================================================
        modules = [self.adc, self.envelope_controller]
        if self.gain_smoother is not None:
            modules.append(self.gain_smoother)

        EPS = 0.00001
        oversampling_factors = np.asarray(
            [module.fs / AUDIO_SAMPLING_RATE for module in modules]
        )

        # check if all the remaining modules have the same clock
        if (
            norm(
                oversampling_factors.reshape(-1, 1)
                - oversampling_factors.reshape(1, -1)
            )
            > EPS
        ):
            raise ValueError(
                "Except for amplifier (which can have a higher clock rate for increasing simulation precision), the remaining modules should have the same clock rate!"
            )

        # check if the sampling rates of all the remaining modules is larger than the amplifier
        # NOTE: using mean is just arbitrary here since the oversampling factors should be equal due to previous check
        if (
            np.mean(oversampling_factors) - self.amplifier.oversampling_factor
        ) / np.min(
            [np.mean(oversampling_factors), self.amplifier.oversampling_factor]
        ) > EPS:
            raise ValueError(
                "all the modules (ADC, Envelope-Controller and Gain-Smoother) should have a lower sampling rate than the amplifier!"
            )

        # how much faster the amplifier module should be simulated so that modules timings match
        # NOTE: oversampling of the modules could be arbitrary (upto the caveat that it may cause frequency shift in the signal).
        # However, the simulation oversampling should be always an inetegr! So that the amplifier module can be fed/run an integer-times faster than the other modules.
        # NOTE: since the next module after the amplifier is the ADC, which is implemented as high-rate ADC + anti-aliasing decimation filter, the amplifier sampling rate
        # should be larger than this high-rate ADC.
        amplifier_simulation_oversampling = (
            self.amplifier.oversampling_factor / ( np.mean(oversampling_factors) * self.adc.oversampling_factor)
        )
        if (
            np.abs(
                amplifier_simulation_oversampling
                - np.round(amplifier_simulation_oversampling)
            )
            / np.mean(
                [
                    amplifier_simulation_oversampling,
                    np.round(amplifier_simulation_oversampling),
                ]
            )
            > EPS
        ):
            raise ValueError(
                "the ratio between oversampling factors of amplifier and the adc module following it should be an integer!"
            )

        self.amplifier_simulation_oversampling = amplifier_simulation_oversampling

        # reset all the modules
        self.reset()

    def reset(self):
        self.amplifier.reset()
        self.adc.reset()
        self.envelope_controller.reset()

        # it may happen that we use or not use any gain smoothing
        if self.gain_smoother is not None:
            self.gain_smoother.reset()

    @property
    def modules_inner_state(self):
        # accumulate all the states from all modules and return it
        modules_inner_state = {
            "amplifier": self.amplifier.state,
            "adc": self.adc.state,
            "envelope_controller": self.envelope_controller.state,
        }

        # check if there is any gain smoother
        if self.gain_smoother is not None:
            modules_inner_state["gain_smoother"] = self.gain_smoother.state
        else:
            modules_inner_state["gain_smoother"] = {}

        return modules_inner_state

    @modules_inner_state.setter
    def modules_inner_state(self, *args, **kwargs):
        raise NotImplementedError(
            "inner states of the internal modules cannot be modified during the simulation!"
        )

    def evolve(
        self,
        audio: np.ndarray,
        audio_sample_rate: float,
        record: bool = False,
        progress_report: bool = False,
    ) -> np.ndarray:
        """this function takes the input signal and simulates amplifier, ADC and AGC.

        Args:
            audio (np.ndarray): input audio signal.
            audio_sample_rate (float): sampling rate of the audio signal.
            record (bool, optional): record the inner states of the sub-modules during the simulation.
            progress_report (bool, optional): show the progress of the simulation.

        Returns:
            np.ndarray: _description_
        """

        adc_out_vec = []
        agc_pga_gain_vec = []
        agc_pga_command_vec = []
        amplifier_out_vec = []
        envelope_vec = []
        gain_smoother_vec = []

        adc_out = 0
        agc_pga_gain = 1.0
        agc_pga_command = 0
        amplifier_out = 0
        envelope = 0
        gain_smoother_out = 0

        # set the iterator
        # if progress_report:
        #     iterator = enumerate(tqdm(audio))
        #     print("\n\n", " simulating the AGC module ".center(120, "+"), "\n")
        # else:
        #     iterator = enumerate(audio)

        if progress_report:
            iterator = enumerate(tqdm(audio))
            print(
                "\n\n",
                "=" * 120,
                "\n",
                " simulating the AGC module ".center(120, "+"),
                "\n",
                "=" * 120,
                "\n",
            )
        else:
            iterator = enumerate(audio)

        # flag for running the amplifier module faster than other modules
        num_samples_fed_to_amplifier = 0
        num_samples_received_from_adc = 0

        for time_idx, sig_in in iterator:
            # input time instant
            time_in = time_idx / audio_sample_rate


            # produce amplifier output
            #! note the old value of agc_pga_command computed in the past clock is used to produce amplifier output and ADC output
            amplifier_out = self.amplifier.evolve(
                audio=sig_in,
                time_in=time_in,
                pga_command=agc_pga_command,
                record=record,
            )

            # NOTE: since amplifier can have a higher clock (for better precision), it should be `self.amplifier_simulation_oversampling` times faster
            num_samples_fed_to_amplifier += 1

            if (
                num_samples_fed_to_amplifier % self.amplifier_simulation_oversampling
            ) > 0:
                # do not run the other modules since they have a smaller sampling rate, i.e., they should be run with a slower clock
                continue

            # produce the ADC output and register the PGA gain used while ADC was quantizing the signal
            adc_out = self.adc.evolve(
                sig_in=amplifier_out,
                time_in=time_in,
                record=record,
            )

            num_samples_received_from_adc += 1
            
            # if adc is in oversampled mode, run the next modules with a lower clock
            if num_samples_received_from_adc % self.adc.oversampling_factor > 0:
                continue
                

            # * record the gain and the gain index that was used at this time slot
            #! Note that as soon as the new clock comes, gain index is updated by envelope controller but that gain index will be used for the current clock
            #! So it does not affect the gain used in ADC
            agc_pga_gain_index_used_in_adc = agc_pga_command
            agc_pga_gain_used_in_adc = self.amplifier.pga_gain_vec[
                agc_pga_gain_index_used_in_adc
            ]

            # * run envelope controller
            #! PGA command is updated and sets the PGA gain value which will appear in the next clock
            agc_pga_command, envelope = self.envelope_controller.evolve(
                sig_in=adc_out, time_in=time_in, record=record
            )

            # compute/update the pga gain value as soon as the rising edge of the clock comes
            #! this new gain will be used in this clock and its effect will appear on ADC signal in the next clock
            #! because during this clock period, ADC is still working to prepare the signal sample
            agc_pga_gain = self.amplifier.pga_gain_vec[agc_pga_command]

            # use the ADC out and the command generated by PGA in gain smoother
            if self.gain_smoother is not None:
                gain_smoother_out = self.gain_smoother.evolve(
                    audio=adc_out,
                    time_in=time_in,
                    pga_gain_index=agc_pga_gain_index_used_in_adc,
                    record=record,
                )

            # save the results
            # NOTE: due to skipping some samples when modules have various sampling rates, 
            # the output of all modules is registered only at lowet sampling rate of all modules
            adc_out_vec.append(adc_out)
            agc_pga_gain_vec.append(agc_pga_gain)
            agc_pga_command_vec.append(agc_pga_command)
            amplifier_out_vec.append(amplifier_out)
            envelope_vec.append(envelope)

            if self.gain_smoother is not None:
                gain_smoother_vec.append(gain_smoother_out)

        adc_out_vec = np.asarray(adc_out_vec)
        agc_pga_command_vec = np.asarray(agc_pga_command_vec)
        agc_pga_gain_vec = np.asarray(agc_pga_gain_vec)
        amplifier_out_vec = np.asarray(amplifier_out_vec)
        envelope_vec = np.asarray(envelope_vec)
        gain_smoother_vec = np.asarray(gain_smoother_vec)

        simulation_state = {
            "agc_pga_command": agc_pga_command_vec,
            "agc_pga_gain": agc_pga_gain_vec,
            "amplifier_output": amplifier_out_vec,
            "envelope": envelope_vec,
            "adc_output": adc_out_vec,
            "gain_smoother_output": gain_smoother_vec,
        }

        # depending on if gain smoother is activated
        return (adc_out_vec, simulation_state) if self.gain_smoother is None else (gain_smoother_vec, simulation_state)
        

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        this is the same as evolve function.
        """
        return self.evolve(*args, **kwargs)

    def __repr__(self) -> str:
        # string representation of the quantization module.
        string = (
            " AGC + ADC module consisting of the following sub-modules ".center(100, "=")
            + "\n\n"
            + str(self.amplifier)
            + "\n\n"
            + str(self.adc)
            + "\n\n"
            + str(self.envelope_controller)
            + "\n\n"
        )

        if self.gain_smoother is not None:
            string += str(self.gain_smoother)

        return string