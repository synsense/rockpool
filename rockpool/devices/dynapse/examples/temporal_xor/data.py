"""
Temporal XOR task sample generator, dataset and dataloaders

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
04/02/2022
"""

from typing import Optional, Tuple

from dataclasses import dataclass

import numpy as np


@dataclass
class SampleGenerator:
    """
    SampleGenerator creates a signal including two pulses and one result representing
    a temporal XOR task. The signal can be temporally shaped using the parameters.
    
    0 ^ 0 = 0 :   
    __    __    ______
      |__|  |__|

    ____________    __
                |__|

    0 ^ 1 = 1 :
             __
    __    __|  |______
      |__|
                 __
    ____________|  |__

    1 ^ 0 = 1 :
       __    
    __|  |__    ______
            |__|
                 __
    ____________|  |__

    1 ^ 1 = 0 :
       __    __
    __|  |__|  |______

    ____________    __
                |__|
    

    :Signal Schudule:                       
                        ________________                                                                ________________
                       |                |                                                              |                |
     margin + delay[0] | pulse_width[0] | min_pulse_delay + delay[1]                    response_delay | response_width | margin + <rest>
    ___________________|                |____________________________                  ________________|                |_________________
                                                                     |                |
                                                                     | pulse_width[1] |
                                                                     |________________|
    :Parameters:        

    :param num_samples: number of samples produced, defaults to 100
    :type num_samples: int, optional
    :param sample_duration: the total sample duration in seconds, defaults to 1.0
    :type sample_duration: float, optional
    :param dt: The time step for quantization, defaults to 1e-3
    :type dt: float, optional
    :param margin: the idle time that no activity is allowed in seconds. Both from the beginning and from the end, defaults to 0.05
    :type margin: float, optional
    :param min_pulse_width: the minimum pulse width allowed in seconds, defaults to 0.1
    :type min_pulse_width: float, optional
    :param min_pulse_delay: the minimum pulse delay between two pulses allowed in seconds, defaults to 0.1
    :type min_pulse_delay: float, optional
    :param response_delay: the response delay past from the last signal to the result in seconds, defaults to 0.05
    :type response_delay: float, optional
    :param response_width: the response pulse width in seconds, defaults to 0.3
    :type response_width: float, optional
    :param sigma: the sigma value used in gaussian filtering (in seconds), defaults to 0.05
    :type sigma: float, optional
    :param amplitude: the absolute amplitude of 2 pulses and the response, defaults to 1.0
    :type amplitude: float, optional
    :param seed: the random seed value to initialize the numpy random state, defaults to None
    :type seed: Optional[int], optional

    :Instance Variables:
    :ivar n_intro: number of time steps until first pulse starts (num_samples,)
    :type n_intro: np.ndarray
    :ivar n_pulse_width: the number of time steps for the first and the seconds pulse stacked together (num_samples, 2)
    :type n_pulse_width: np.ndarray
    :ivar n_pulse_delay: effective pulse delay in time steps from the end of the first pulse till the beginning of the second pulse (num_samples,)
    :type n_pulse_delay: np.ndarray
    :ivar n_response_delay: the number of timesteps passed from the end of the second pulse till the beginning of the response (1,)
    :type n_response_delay: int
    :ivar n_response_width: the response pulse width in number of timesteps (1,)
    :type n_response_width: int
    :ivar n_sample_duration: the total sample duration in number of timesteps (1,)
    :type n_sample_duration: int
    :ivar n_outro: the number of timesteps from the end of the response till the end of the sampling duration (num_samples,)
    :type n_outro: np.ndarray
    :ivar sign: the signs of 2 of the pulses and 1 response stacked together (num_samples,3)
    :type sign: np.ndarray
    """

    num_samples: int = 100
    sample_duration: float = 1.0
    dt: float = 1e-3
    margin: float = 0.05
    min_pulse_width: float = 0.1
    min_pulse_delay: float = 0.1
    response_delay: float = 0.05
    response_width: float = 0.3
    sigma: float = 0.05
    amplitude: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self):
        """
        __post_init__ randomly schedule the pulse widths and start times, then quantize the timing
        in order to be able to create discrete signal arrays.
        """
        n_steps = lambda arg: np.around(arg / self.dt).astype(int)
        self.n_sigma = n_steps(self.sigma)

        # Scheduling
        np.random.seed(self.seed)
        pulse_width, delay = self.random_schedule()

        # Idle start
        self.n_intro = n_steps(self.margin + delay[:, 0])  # (num_samples,)

        # Pulses
        self.n_pulse_width = n_steps(pulse_width)  # (num_samples, 2)
        self.n_pulse_delay = n_steps(
            self.min_pulse_delay + delay[:, 1]
        )  # (num_samples,)

        # Response
        self.n_response_delay = n_steps(self.response_delay)  # (1,)
        self.n_response_width = n_steps(self.response_width)  # (1,)

        # Idle end
        self.n_sample_duration = n_steps(self.sample_duration)  # (1,)
        self.n_outro = self.n_sample_duration - (  # (num_samples,)
            self.n_intro
            + self.n_pulse_width.sum(axis=1)
            + self.n_pulse_delay
            + self.n_response_delay
            + self.n_response_width
        )

        # Smoothing Filter
        self.filter = self.gaussian_filter()
        # XOR Logic Represented
        self.logic = self.logical_xor()  # (num_samples,3)

    def logical_xor(self) -> np.ndarray:
        """
        logic builts the logic behind the temporal xor task. It creates a batch of instances
        
        :return: signed xor sample with 2 inputs and 1 output (num_samples, 3)
        :rtype: np.ndarray
        """

        logits = np.random.rand(2, self.num_samples) > 0.5
        xor_result = np.logical_xor(logits[0], logits[1])
        signed_xor = np.vstack((logits, xor_result)).T * 2 - 1
        return signed_xor

    def gaussian_filter(self) -> np.ndarray:
        """
        gaussian_filter creates a gaussian filter given the sigma value for smoothing purposes
        
        .. math ::

            \\dfrac{1}{\\sqrt{2\\pi \\sigma^{2}}} \\text{exp} \\left( -\\dfrac{(x-\\mu)^{2}}{2\\sigma^{2}}   \\right)

        :return: a padded signal length gaussian filter (n_sample_duration,)
        :rtype: np.ndarray
        """
        base = lambda s: np.linspace(-s / 2, s / 2, s)
        centered_time_base = base(self.n_sample_duration)

        filter = (1.0 / (self.n_sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * np.square(centered_time_base / self.n_sigma)
        )

        return filter

    def random_schedule(self) -> Tuple[np.ndarray]:
        """
        random_schedule randomly determines the widths and start times of the input pulses

        :return: pulse_width, delay
            :pulse_width: the widths of two pulses in seconds
            :delay: the delays occur prior to the pulses
        :rtype: Tuple[np.ndarray]
        """
        ## NOMINAL RESPONSE DURATION ##
        #                  ________________
        #                 |                |
        #  response_delay | response_width | margin
        # ________________|                |________

        response_duration = self.response_delay + self.response_width + self.margin

        ## NOMINAL INPUT DURATION ##
        #                     ________________                          ________________
        #                    |                |                        |                |
        #  margin + delay[0] | pulse_width[0] | pulse_delay + delay[1] | pulse_width[1] |
        # ___________________|                |________________________|                |

        input_duration = self.sample_duration - response_duration
        max_signal_width = (input_duration - self.min_pulse_delay) / 2.0

        # Pulse Generation
        pulse_width = np.random.uniform(
            low=self.min_pulse_width, high=max_signal_width, size=(self.num_samples, 2)
        )

        signal_duration = (
            self.margin + pulse_width[:, 0] + self.min_pulse_delay + pulse_width[:, 1]
        )  # (num_samples,)

        # Pulse 1
        p1_freedom = input_duration - signal_duration
        p1_delay = np.random.uniform(low=0, high=p1_freedom, size=self.num_samples)

        # Pulse 2
        p2_freedom = p1_freedom - p1_delay
        p2_delay = np.random.uniform(low=0, high=p2_freedom, size=self.num_samples)

        # Signal Delay
        delay = np.stack((p1_delay, p2_delay)).T

        return pulse_width, delay

    def input_signal(
        self, intro: int, pulse_width: np.ndarray, pulse_delay: int, sign: np.ndarray,
    ) -> np.ndarray:
        """
        input_signal creates the input signal with two square-wave pulses

                ________________               ________________
               |                |             |                |
         intro | pulse_width[0] | pulse_delay | pulse_width[1] |
        _______|                |_____________|                |


        :param intro: the idle time passed until the first pulse in number of timesteps (1,)
        :type intro: int
        :param pulse_width: the pulse widths of the square waves in number of timesteps (2,)
        :type pulse_width: np.ndarray
        :param pulse_delay: the time interval between the pulses in number of timesteps (1,)
        :type pulse_delay: int
        :param sign: array indicating the directions of the waves (2,)
        :type sign: np.ndarray
        :return: a temporal xor signal
        :rtype: np.ndarray
        """
        # Idles
        intro = np.zeros(intro)
        pulse_delay = np.zeros(pulse_delay)

        # Pulses
        pulse_1 = np.full(pulse_width[0], sign[0] * self.amplitude)
        pulse_2 = np.full(pulse_width[1], sign[1] * self.amplitude)

        signal = np.concatenate((intro, pulse_1, pulse_delay, pulse_2))
        return signal

    def response_signal(self, outro: int, sign: int) -> np.ndarray:
        """
        response_signal creates the xor response signal with one square-wave pulse
                         ________________
                        |                |
         response_delay | response_width | outro
        ________________|                |_________        

        :param outro: the idle time passed after the response wave ends in number of timesteps
        :type outro: int
        :param sign: the direction of the response in number of timesteps
        :type sign: int
        :return: a temporal xor gate result
        :rtype: np.ndarray
        """
        # Idles
        response_delay = np.zeros(self.n_response_delay)
        outro = np.zeros(outro)

        # Pulses
        pulse_response = np.full(self.n_response_width, sign * self.amplitude)

        signal = np.concatenate((response_delay, pulse_response, outro))
        return signal

    def pad_sample(
        self, input_signal: np.ndarray, response_signal: np.ndarray
    ) -> np.ndarray:
        """
        pad_sample applies padding to the input signal and the response signal in order to have equal length signals.
            
            Padded input and the response signal
                     __
            __    __|  |______
              |__|
                         __
            ____________|  |__
        
        :param input_signal: the input xor signal
        :type input_signal: np.ndarray
        :param response_signal: the xor gate response signal
        :type response_signal: np.ndarray
        :return: a training instance including an input signal and the expected response (2,n_sample_length) 
        :rtype: np.ndarray
        """
        # Get the paddings
        input_pad = np.zeros_like(input_signal)
        response_pad = np.zeros_like(response_signal)

        # Get the signals
        input_signal = np.concatenate((input_signal, response_pad))
        response_signal = np.concatenate((input_pad, response_signal))

        # Stack together
        sample = np.stack((input_signal, response_signal))
        return sample

    def raw_sample(self, idx: int) -> np.ndarray:
        """
        sample creates the input and response signals using the generated schedule

        :param idx: the sample index
        :type idx: int
        :return: a training instance including an input signal and the expected response (2,n_sample_length) 
        :rtype: np.ndarray
        """
        sign = self.logic[idx]

        # Create signals
        input_signal = self.input_signal(
            self.n_intro[idx],
            self.n_pulse_width[idx],
            self.n_pulse_delay[idx],
            sign[0:2],
        )
        response_signal = self.response_signal(self.n_outro[idx], sign[2])

        # Do the padding
        _sample = self.pad_sample(input_signal, response_signal)
        return _sample

    def batch(self, offset: int = 0, size: Optional[int] = None) -> np.ndarray:
        """
        batch creates a batch of raw samples and then applies gaussian filter
        
        :param offset: the read index start offset, defaults to 0
        :type offset: int, optional
        :param size: the size of the sample batch, should be smaller than or equal to the number of samples, defaults to None
        :type size: Optional[int], optional
        :return: a batch of processed and smoothed samples (batch, subset, length)
        :rtype: np.ndarray
        """
        if size is None:
            size = self.num_samples

        # Get a batch of samples
        _batch = np.stack(
            [self.raw_sample(idx) for idx in range(offset, offset + size)]
        )
        _batch = _batch.reshape(-1, self.n_sample_duration)

        # Convolve each element
        _conv = lambda _b: np.convolve(_b, self.filter, "same")
        _batch = np.apply_along_axis(_conv, 1, _batch)

        # Return the same shape processed signal batch
        _batch = _batch.reshape(size, 2, self.n_sample_duration)
        return _batch
