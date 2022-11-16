"""
Temporal XOR task sample generator, dataset and dataloaders

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
04/02/2022
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import os
from dataclasses import dataclass
import numpy as np
import h5py
from tonic.dataset import Dataset

import matplotlib.pyplot as plt


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

    def __post_init__(self) -> None:
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
        self,
        intro: int,
        pulse_width: np.ndarray,
        pulse_delay: int,
        sign: np.ndarray,
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


class TemporalXORData:
    """
    TemporalXORData is the data management module for temporal xor task. It creates
    train, validation and test sets of the data samples and store in disk

    :Parameters:

    :param train_val_test: train, validation, and test set split ratio, defaults to [0.88, 0.1, 0.02]
    :type train_val_test: np.ndarray, optional

    :Instance Variables:

    :ivar sample_generator: sample generator object holding a virtual dataset with all the sample properties
    :type sample_generator: SampleGenerator
    :ivar num_samples: number of samples produced
    :type num_samples: int
    :ivar n_sample_duration: discrete sample duration, the number of timesteps
    :type n_sample_duration: int
    :ivar num_train_samples: the number of training samples
    :type num_train_samples: int
    :ivar num_val_samples: the number of validation samples
    :type num_val_samples: int
    :ivar num_test_samples: the number of test samples
    :type num_test_samples: int
    """

    def __init__(
        self,
        *args,
        train_val_test: np.ndarray = [0.88, 0.1, 0.02],
        **kwargs,
    ) -> None:
        """
        __init__ Initialize ``TemporalXORData`` module. Parameters are explained in the class docstring.
        """

        # Sample Generator
        self.sample_generator = SampleGenerator(*args, **kwargs)
        self.num_samples = self.sample_generator.num_samples
        self.n_sample_duration = self.sample_generator.n_sample_duration
        n_sample = lambda _r: int(np.floor(self.num_samples * _r))

        # Special setter with error checking
        self.train_val_test = train_val_test

        # Train / Val / Test samples
        self.num_train_samples = n_sample(self.train_val_test[0])
        self.num_val_samples = n_sample(self.train_val_test[1])
        self.num_test_samples = (
            self.num_samples - self.num_train_samples - self.num_val_samples
        )

    def fill_hdf5(self, hdf5_path: str) -> None:
        """
        fill_hdf5 initiate and fill an hdf5 database in one shot
        Note that to generate and fill hdf5 dataset in one shot, a bigger memory space is required!

        :param hdf5_path: the filepath for hdf to be created!
        :type hdf5_path: str
        """

        with h5py.File(hdf5_path, "a") as hdf:
            for _subset in self.subsets:
                group = hdf.create_group(_subset)

                # (subset, batch, length) -> (subset, batch, length)
                _input, _response = self.sample_generator.batch(
                    self.offset[_subset], self.samples[_subset]
                ).transpose(1, 0, 2)

                # Fill datasets
                group.create_dataset(self.datasets[0], data=_input)
                group.create_dataset(self.datasets[1], data=_response)

    def batch_size_array(self, subset: str, memory_batch: int) -> np.ndarray:
        """
        batch_size_array calls the `batch_split()` given the desired subset : train, val or test

        :param subset: train, val or test
        :type subset: str
        :param memory_batch: the maximum number of samples to store at any time in memory (applies when disk storage is used). If None, then all samples are created and stored in memory at once
        :type memory_batch: int
        :return: batch size array, elements indicating the number of samples genrated in each iteration
        :rtype: np.ndarray
        """
        num_samples = self.samples[subset]

        if memory_batch >= num_samples:
            logging.info(
                f"Memory batch is greater than the number of samples generated! : {memory_batch} > {num_samples}"
            )
            return np.array([num_samples])

        return self.batch_split(num_samples, memory_batch)

    def setup_hdf5(self, hdf5_path: str) -> None:
        """
        setup_hdf5 initiate an empty hdf5 database

        :param hdf5_path: the filepath for hdf to be created!
        :type hdf5_path: str
        """

        with h5py.File(hdf5_path, "a") as hdf:
            for _subset in self.subsets:
                group = hdf.create_group(_subset)
                for _dataset in self.datasets:
                    group.create_dataset(
                        _dataset,
                        shape=(0, self.n_sample_duration),
                        chunks=True,
                        maxshape=(None, self.n_sample_duration),
                    )

    def fill_batch(
        self, hdf: h5py.File, subset: str, offset: int, batch_iterations: np.ndarray
    ) -> None:
        """
        fill_batch fills an hdf database file in batch append mode

        :param hdf: hdf file opened in append mode
        :type hdf: h5py.File
        :param subset: the subset to fill "train", "val", or "test"
        :type subset: str
        :param offset: the read index start offset
        :type offset: int
        :param batch_iterations: a list of a number of samples to be generated and written to disk in each iteration like [100,100,88]
        :type batch_iterations: np.ndarray
        """

        def append(_dataset: str, data: np.ndarray) -> None:
            """
            append merges the data given to the existing dataset.
            Note that the first dimension of the dataset would be extended but the second dimensions should match.

            :param _dataset: the name of the dataset
            :type _dataset: str
            :param data: the data to be appended to the existing dataset
            :type data: np.ndarray
            """
            _shape = hdf[subset][_dataset].shape[0]
            _extend = data.shape[0]

            # Resize and fill
            hdf[subset][_dataset].resize(_shape + _extend, axis=0)
            hdf[subset][_dataset][-_extend:] = data

        # Fill in batch mode
        for batch_size in batch_iterations:

            # (batch, subset, length)
            data = self.sample_generator.batch(offset, batch_size)
            offset += batch_size

            # (subset, batch, length)
            _input, _response = data.transpose(1, 0, 2)
            append("input", _input)
            append("response", _response)

    def save_disk(self, hdf5_path: str, memory_batch: Optional[int] = None) -> None:
        """
        save_disk creates a database consisting of train, validation and tests sets of temporal xor samples
        with input and response signals, then save this to the disk. If free memory space is small, then
        memory batch option provides an opportunity to create the samples in batches then saving to the disk
        in append mode

        :param hdf5_path: the file path for the newly generated hdf5 database
        :type hdf5_path: str
        :param memory_batch: the maximum number of samples to store at any time in memory (applies when disk storage is used). If None, then all samples are created and stored in memory at once, defaults to None
        :type memory_batch: Optional[int], optional
        :raises OSError: File already exists!
        """

        # Check if exist
        if os.path.exists(hdf5_path):
            raise OSError(f"File already exists! : {hdf5_path}")

        else:
            dirname = os.path.dirname(hdf5_path)
            os.makedirs(dirname, exist_ok=True)

        with h5py.File(hdf5_path, "w") as hdf:
            for key, value in self.sample_generator.__dict__.items():
                if value is not None:
                    hdf.attrs[key] = value
            hdf.attrs["train_val_test"] = self.train_val_test

        # Fill
        if memory_batch is not None:
            self.setup_hdf5(hdf5_path)
            with h5py.File(hdf5_path, "a") as hdf:
                for _sub in self.subsets:
                    self.fill_batch(
                        hdf,
                        _sub,
                        self.offset[_sub],
                        self.batch_size_array(_sub, memory_batch),
                    )
        else:
            self.fill_hdf5(hdf5_path)

    @staticmethod
    def batch_split(num_samples: int, batch_size: int) -> np.ndarray:
        """
        batch_split  creates a batch size array when memory batching is applied.
        i.e. [100, 100, 88] means that in the first iteration, 100 samples
        are fitted to the memory, in second 100, in the last 88.

        :param num_samples: the total number of samples
        :type num_samples: int
        :batch_size: the desired batch_size
        :type batch_size: int
        :return: batch size array, elements indicating the number of samples genrated in each iteration
        :rtype: np.ndarray
        """
        n_iteration = num_samples // batch_size  # 288 // 100 = 2
        iterations = np.full(n_iteration, batch_size)  # [100, 100]
        last = num_samples - n_iteration * batch_size
        if last != 0:
            iterations = np.concatenate((iterations, [last]))  # [100, 100, 88]
        return iterations

    @property
    def train_offset(self) -> int:
        """
        train_offset is the starting index for fetching training samples from the virtual samples stored in SampleGenerator
        """
        return 0

    @property
    def val_offset(self) -> int:
        """
        val_offset is the starting index for fetching validation samples from the virtual samples stored in SampleGenerator
        """
        return self.num_train_samples

    @property
    def test_offset(self) -> int:
        """
        test_offset is the starting index for fetching test samples from the virtual samples stored in SampleGenerator
        """
        return self.num_train_samples + self.num_val_samples

    @property
    def offset(self) -> Dict[str, int]:
        """
        offset returns a dictionary combining the train, val and test offsets
        """
        return {
            "train": self.train_offset,
            "val": self.val_offset,
            "test": self.test_offset,
        }

    @property
    def samples(self) -> Dict[str, int]:
        """
        samples returns a dictionary combining the number of train, val and test samples
        """
        return {
            "train": self.num_train_samples,
            "val": self.num_val_samples,
            "test": self.num_test_samples,
        }

    @property
    def subsets(self) -> List[str]:
        """
        subsets returns a list of subsets in the database
        """
        return ["train", "val", "test"]

    @property
    def datasets(self) -> List[str]:
        """
        subsets returns a list of datasets in the subsets of the database
        """
        return ["input", "response"]

    @property
    def train_val_test(self) -> np.ndarray:
        """
        train_val_test holds the train, validation, and test set split ratio, like [0.88, 0.1, 0.02]
        """
        return self._train_val_test

    @train_val_test.setter
    def train_val_test(self, value):
        """
        train_val_test setter checks if train_val_test ratio split is structured properly,
        If OK, then set self._train_val_test

        :param value: train, validation, and test set split ratio
        :type value: np.ndarray
        :raises ValueError: train_val_test includes more or less than 3 elements
        :raises ValueError: any of the values less than 0
        :raises ValueError: train_val_test does not add up to a value between 0 and 1
        """

        value = np.array(value)

        if len(value) != 3:
            raise ValueError("train_val_test tuple should include 3 numbers!")

        if (value <= 0).any():
            raise ValueError("All train_val_test values must be greater than 0!")

        split_sum = np.sum(value)
        if split_sum > 1.0 or split_sum < 0:
            raise ValueError("Split ratio should add up to a number between 0 and 1!")

        self._train_val_test = value


class TemporalXOR(Dataset):
    """
    TemporalXOR is the Tonic dataset handler for toy temporal xor task. It creates a database with desired number of samples
    and saves it to the disk if the database is not present at the system, or if different sample properties are desired.

    :Parameters:

    :param save_to: location to save files to on disk
    :type save_to: str
    :param train: select the training set or not (least prior one when it comes to conflicts, default option), defaults to None
    :type train: Optional[bool], optional
    :param val: select the val set or not (have priority over train, not over test), defaults to None
    :type val: Optional[bool], optional
    :param test: select the test set or not (have priority over all), defaults to None
    :type test: Optional[bool], optional
    :param transform: a callable of transforms to apply to the data, defaults to None
    :type transform: Optional[Callable], optional
    :param target_transform: a callable of transforms to apply to the targets/labels, defaults to None
    :type target_transform: Optional[Callable], optional
    :param memory_batch: the maximum number of samples to store at any time in memory. Use if data files will be re-created and the number of samples are too big to fit into the memory. If None, then all samples are created and stored in memory at once, defaults to None
    :type memory_batch: Optional[int], optional

    :Instance Variables:

    :ivar subset: the name of the subset in database : "train", "val", or "test"
    :type subset: str
    :ivar data_file: h5py file opened for easy access
    :type data_file: h5py.File
    """

    extension = ".hdf5"
    filename = "txor" + extension

    def __init__(
        self,
        save_to: str,
        train: Optional[bool] = None,
        val: Optional[bool] = None,
        test: Optional[bool] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        memory_batch: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:

        super(TemporalXOR, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        if self._check_exists() and (len(args) > 0 or len(kwargs) > 0):
            # If any input found in *arg or in **kwargs, then delete the existing database and re-create
            os.remove(os.path.join(self.location_on_system, self.filename))

        if not self._check_exists():
            # Create & save
            logging.warning(
                f"The {self.filename} database in {self.location_on_system} is missing. It is being created just for once!."
            )
            TemporalXORData(*args, **kwargs).save_disk(
                os.path.join(self.location_on_system, self.filename), memory_batch
            )

        self.subset = self.select_set(train, val, test)

        # Open data file
        self.data_file = h5py.File(
            os.path.join(self.location_on_system, self.filename), "r"
        )

    def select_set(
        self, train: Optional[bool], val: Optional[bool], test: Optional[bool]
    ) -> str:
        """
        select_set selects the subset "train", "val" or "test"

        :param train: select the training set or not (least prior one when it comes to conflicts, default option), defaults to None
        :type train: Optional[bool], optional
        :param val: select the val set or not (have priority over train, not over test), defaults to None
        :type val: Optional[bool], optional
        :param test: select the test set or not (have priority over all), defaults to None
        :type test: Optional[bool], optional
        :return: the name of the subset in database : "train", "val", or "test"
        :rtype: str
        """
        _set = None

        if test:
            _set = "test"
        elif val:
            _set = "val"
        elif train or (train is None and val is None and test is None):
            _set = "train"

        logging.info(f"Temporal xor {_set} set is ready!")
        return _set

    def __getitem__(self, index: int) -> Tuple[np.ndarray]:
        """
        __getitem__ fetches a data sample from the database, applies transforms and returns
        the data sample and the target stacked together.

        :param index: the index of the data sample inquired
        :type index: int
        :return: a sample instance including the data and the target (2, n_sample_duration)
        :rtype: Tuple[np.ndarray]
        """

        # Fetch Data
        data = np.array(self.data_file[self.subset]["input"][index])
        target = np.array(self.data_file[self.subset]["response"][index])

        data = np.expand_dims(data, -1)
        target = np.expand_dims(target, -1)

        # Apply transforms
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Return
        return data, target

    def plot(self, index: int) -> None:
        """
        plot visualizes the data and target nicely and simply

        :param index: the index of the data sample inquired
        :type index: int
        """

        # Fetching
        data, target = self.__getitem__(index)
        duration = self.data_file.attrs["sample_duration"]
        dt = self.data_file.attrs["dt"]
        time_base = np.arange(0, duration, dt)

        # Plotting
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        ax1.set_title(
            f"XOR {self.subset} sample [{index}/{self.__len__()-1}]: data & target"
        )
        ax1.plot(time_base, data)
        ax2.plot(time_base, target)
        ax2.set_xlabel("Time (s)")
        plt.show()

    def __len__(self) -> int:
        """
        __len__ returns the length of the dataset
        """
        return len(self.data_file[self.subset]["response"])

    def _check_exists(self) -> bool:
        """
        _check_exists checks if the hdf5 data file present in the expected location
        """
        return (
            self._is_file_present()
            and self._folder_contains_at_least_n_files_of_type(1, self.extension)
        )
