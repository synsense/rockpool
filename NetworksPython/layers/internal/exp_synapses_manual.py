###
# exp_synapses_manual.py - Class implementing a spike-to-current layer with exponential synapses
###


# - Imports
from typing import Union
import numpy as np
from scipy.signal import fftconvolve
import torch

from ...timeseries import TSContinuous, TSEvent
from ..layer import Layer

from typing import Optional, Union, Tuple, List

from warnings import warn

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = ["FFExpSyn"]

# - Absolute tolerance, e.g. for comparing float values
tol_abs = 1e-9


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


## - FFExpSyn - Class: define an exponential synapse layer (spiking input)
class FFExpSyn(Layer):
    """ FFExpSyn - Class: define an exponential synapse layer (spiking input)
    """

    ## - Constructor
    def __init__(
        self,
        weights: Union[np.ndarray, int] = None,
        bias: np.ndarray = 0,
        dt: float = 0.0001,
        noise_std: float = 0,
        tau_syn: float = 0.005,
        name: str = "unnamed",
        add_events: bool = True,
    ):
        """
        FFExpSyn - Construct an exponential synapse layer (spiking input)

        :param weights:             np.array MxN weight matrix
                                int Size of layer -> creates one-to-one conversion layer
        :param dt:             float Time step for state evolution
        :param noise_std:       float Std. dev. of noise added to this layer. Default: 0

        :param tau_syn:         float Output synaptic time constants. Default: 5ms
        :param synapse_eq:      Brian2.Equations set of synapse equations for receiver. Default: exponential
        :param integrator_name:   str Integrator to use for simulation. Default: 'exact'

        :param name:         str Name for the layer. Default: 'unnamed'

        :add_events:            bool     If during evolution multiple input events arrive during one
                                         time step for a channel, count their actual number instead of
                                         just counting them as one.
        """

        # - Provide default weight matrix for one-to-one conversion
        if isinstance(weights, int):
            weights = np.identity(weights, "float")

        # - Check dt
        if dt is None:
            dt = tau_syn / 10

        # - Call super constructor
        super().__init__(
            weights=weights, dt=dt, noise_std=np.asarray(noise_std), name=name
        )

        # - Parameters
        self.tau_syn = tau_syn
        self.bias = bias
        self.add_events = add_events

        # - set time and state to 0
        self.reset_all()

        # - Objects for training
        self._xtx = None
        self._xty = None

    def _prepare_input(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> (np.ndarray, int):
        """
        _prepare_input - Sample input and return as raster.

        :param ts_input:      TimeSeries TxM or Tx1 Input signals for this layer
        :param duration:    float Duration of the desired evolution, in seconds
        :param num_timesteps int Number of evolution time steps

        :return:
            mnInput:          ndarray Raster containing spike info
            num_timesteps:    ndarray Number of evlution time steps
        """
        if num_timesteps is None:
            # - Determine num_timesteps
            if duration is None:
                # - Determine duration
                assert (
                    ts_input is not None
                ), "Layer {}: One of `num_timesteps`, `ts_input` or `duration` must be supplied".format(
                    self.name
                )

                if ts_input.periodic:
                    # - Use duration of periodic TimeSeries, if possible
                    duration = ts_input.duration

                else:
                    # - Evolve until the end of the input TImeSeries
                    duration = ts_input.t_stop - self.t + self.dt
                    assert duration > 0, (
                        "Layer {}: Cannot determine an appropriate evolution duration.".format(
                            self.name
                        )
                        + "`ts_input` finishes before the current "
                        "evolution time."
                    )
            # - Discretize duration wrt self.dt
            num_timesteps = int(np.floor((duration + tol_abs) / self.dt))
        else:
            assert isinstance(
                num_timesteps, int
            ), "Layer `{}`: num_timesteps must be of type int.".format(self.name)

        if ts_input is not None:
            # Extract spike data from the input variable
            spike_raster = ts_input.raster(
                dt=self.dt,
                t_start=self.t,
                num_timesteps=num_timesteps,
                channels=np.arange(self.size_in),
                add_events=self.add_events,
            ).astype(float)

        else:
            spike_raster = np.zeros((num_timesteps, self.size_in))

        return spike_raster, num_timesteps

    ### --- State evolution

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSEvent  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSContinuous  output spike series

        """

        # - Prepare weighted input signal
        inp_raster, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )
        weighted_input = inp_raster @ self.weights

        # - Time base
        time_base = (np.arange(num_timesteps + 1) + self._timestep) * self.dt

        if self.noise_std > 0:
            # - Add a noise trace
            # - Noise correction is slightly different than in other layers
            noise = (
                np.random.randn(*weighted_input.shape)
                * self.noise_std
                * np.sqrt(2 * self.dt / self.tau_syn)
            )
            noise[0, :] = 0  # Make sure that noise trace starts with 0
            weighted_input += noise

        # Add current state to input
        weighted_input[0, :] += self._state_no_bias.copy() * np.exp(
            -self.dt / self.tau_syn
        )

        # - Define exponential kernel
        kernel = np.exp(-np.arange(num_timesteps + 1) * self.dt / self.tau_syn)
        # - Make sure spikes only have effect on next time step
        kernel = np.r_[0, kernel]

        # - Apply kernel to spike trains
        filtered = np.zeros((num_timesteps + 1, self.size))
        for channel, events in enumerate(weighted_input.T):
            conv = fftconvolve(events, kernel, "full")
            conv_short = conv[: time_base.size]
            filtered[:, channel] = conv_short

        # - Update time and state
        self._timestep += num_timesteps
        self._state_no_bias = filtered[-1]

        # - Output time series with output data and bias
        return TSContinuous(time_base, filtered + self.bias, name="Receiver current")

    def evolve_train(
        self,
        ts_target: TSContinuous,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        regularize: float = 0,
        learning_rate: float = 0.01,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        evolve : Function to evolve the states of this layer given an input

        :param ts_target:        TSContinuous  Target time series
        :param tsSpkInput:      TSEvent  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param regularize:     float    Regularization parameter
        :param learning_rate:   flaot    Factor determining scale of weight increments at each step
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSContinuous  output spike series

        """

        # - Prepare input signal
        num_timesteps = int(np.round(ts_target.duration / self.dt))
        inp_raster, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Time base
        time_base = (np.arange(num_timesteps + 1) + self._timestep) * self.dt

        # - Define exponential kernel
        kernel = np.exp(-(np.arange(num_timesteps) * self.dt) / self.tau_syn)
        # - Make sure spikes only have effect on next time step
        kernel = np.r_[0, kernel]

        # Empty input array with additional dimension for training biases
        inp = np.zeros((np.size(time_base), self.size_in + 1))
        inp[:, -1] = 1

        # - Apply kernel to spike trains and add filtered trains to input array
        for channel, events in enumerate(inp_raster.T):
            inp[:, channel] = fftconvolve(events, kernel, "full")[: time_base.size]

        # - Evolution:
        weighted = inp[:, :-1] @ self.weights
        out = weighted + self.bias

        # - Update time and state
        self._timestep += num_timesteps

        ## -- Training
        # - Prepare target data
        target = ts_target(time_base)

        # - Make sure no nan is in target, as this causes learning to fail
        assert not np.isnan(
            target
        ).any(), "Layer `{}`: nan values have been found in target (where: {})".format(
            self.name, np.where(np.isnan(target))
        )

        # - Check target dimensions
        if target.ndim == 1 and self.size == 1:
            target = target.reshape(-1, 1)

        assert (
            target.shape[-1] == self.size
        ), "Layer `{}`: Target dimensions ({}) does not match layer size ({})".format(
            self.name, target.shape[-1], self.size
        )

        # - Weight update
        # mfUpdate = inp.T @ (target - out)
        # print(np.linalg.norm(target-out))
        # # Normalize learning rate by number of inputs
        # learning_rate /= (self.size_in * inp.shape[0] * vfG)
        # self.weights += learning_rate * (mfUpdate[:-1]) - regularize * self.weights
        # self.bias += learning_rate * (mfUpdate[-1]) - regularize * self.bias

        xtx = inp.T @ inp
        xty = inp.T @ target
        new_weights = np.linalg.solve(xtx + regularize * np.eye(inp.shape[1]), xty)
        print(np.linalg.norm(target - out))
        self.weights = (self.weights + learning_rate * new_weights[:-1]) / (
            1.0 + learning_rate
        )
        self.bias = (self.bias + learning_rate * new_weights[-1]) / (
            1.0 + learning_rate
        )

        # - Output time series with output data and bias
        return TSContinuous(time_base, out, name="Receiver current")

    def train_rr(
        self,
        ts_target: TSContinuous,
        ts_input: TSEvent = None,
        regularize: float = 0,
        is_first: bool = True,
        is_last: bool = False,
        store_states: bool = True,
        train_biases: bool = True,
    ):

        """
        train_rr - Train self with ridge regression over one of possibly
                   many batches. Use Kahan summation to reduce rounding
                   errors when adding data to existing matrices from
                   previous batches.
        :param ts_target:        TimeSeries - target for current batch
        :param ts_input:         TimeSeries - input to self for current batch
        :regularize:           float - regularization for ridge regression
        :is_first:                bool - True if current batch is the first in training
        :is_last:                bool - True if current batch is the last in training
        :store_states:           bool - Include last state from previous training and store state from this
                                       traning. This has the same effect as if data from both trainings
                                       were presented at once.
        :param train_biases:    bool - If True, train biases as if they were weights
                                       Otherwise present biases will be ignored in
                                       training and not be changed.
        """

        # - Discrete time steps for evaluating input and target time series
        num_timesteps = int(np.round(ts_target.duration / self.dt))
        time_base = self._gen_time_trace(ts_target.t_start, num_timesteps)

        if not is_last:
            # - Discard last sample to avoid counting time points twice
            time_base = time_base[:-1]

        # - Make sure time_base does not exceed ts_target
        time_base = time_base[time_base <= ts_target.t_stop]

        # - Prepare target data
        target = ts_target(time_base)

        # - Make sure no nan is in target, as this causes learning to fail
        assert not np.isnan(
            target
        ).any(), "Layer `{}`: nan values have been found in target (where: {})".format(
            self.name, np.where(np.isnan(target))
        )

        # - Check target dimensions
        if target.ndim == 1 and self.size == 1:
            target = target.reshape(-1, 1)

        assert (
            target.shape[-1] == self.size
        ), "Layer `{}`: Target dimensions ({}) does not match layer size ({})".format(
            self.name, target.shape[-1], self.size
        )

        # - Prepare input data
        input_size = self.size_in + int(train_biases)
        # Empty input array with additional dimension for training biases
        inp = np.zeros((np.size(time_base), input_size))
        if train_biases:
            inp[:, -1] = 1

        # - Generate spike trains from ts_input
        if ts_input is None:
            # - Assume zero input
            print(
                "Layer `{}`: No ts_input defined, assuming input to be 0.".format(
                    self.name
                )
            )

        else:
            # - Get data within given time range
            event_times, event_channels = ts_input(
                t_start=time_base[0], t_stop=time_base[-1]
            )

            # - Make sure that input channels do not exceed layer input dimensions
            try:
                assert (
                    np.amax(event_channels) <= self.size_in - 1
                ), "Layer `{}`: Number of input channels exceeds layer input dimensions.".format(
                    self.name
                )
            except ValueError as e:
                # - No events in input data
                if event_channels.size == 0:
                    print("Layer `{}`: No input spikes for training.".format(self.name))
                else:
                    raise e

            # Extract spike data from the input
            spike_raster = (
                ts_input.raster(
                    dt=self.dt,
                    t_start=time_base[0],
                    num_timesteps=time_base.size,
                    channels=np.arange(self.size_in),
                    add_events=self.add_events,
                )
            ).astype(float)

            if store_states and not is_first:
                try:
                    # - Include last state from previous batch
                    spike_raster[0, :] += self._training_state
                except AttributeError:
                    pass

            # - Define exponential kernel
            kernel = np.exp(-(np.arange(time_base.size - 1) * self.dt) / self.tau_syn)
            # - Make sure spikes only have effect on next time step
            kernel = np.r_[0, kernel]

            # - Apply kernel to spike trains and add filtered trains to input array
            for channel, events in enumerate(spike_raster.T):
                inp[:, channel] = fftconvolve(events, kernel, "full")[: time_base.size]

        # - For first batch, initialize summands
        if is_first:
            # Matrices to be updated for each batch
            self._xty = np.zeros((input_size, self.size))  # inp.T (dot) target
            self._xtx = np.zeros((input_size, input_size))  # inp.T (dot) inp
            # Corresponding Kahan compensations
            self.kahan_comp_xty = np.zeros_like(self._xty)
            self.kahan_comp_xtx = np.zeros_like(self._xtx)

        # - New data to be added, including compensation from last batch
        #   (Matrix summation always runs over time)
        upd_xty = inp.T @ target - self.kahan_comp_xty
        upd_xtx = inp.T @ inp - self.kahan_comp_xtx

        if not is_last:
            # - Update matrices with new data
            new_xty = self._xty + upd_xty
            new_xtx = self._xtx + upd_xtx
            # - Calculate rounding error for compensation in next batch
            self.kahan_comp_xty = (new_xty - self._xty) - upd_xty
            self.kahan_comp_xtx = (new_xtx - self._xtx) - upd_xtx
            # - Store updated matrices
            self._xty = new_xty
            self._xtx = new_xtx

            if store_states:
                # - Store last state for next batch
                if train_biases:
                    self._training_state = inp[-1, :-1].copy()
                else:
                    self._training_state = inp[-1, :].copy()

        else:
            # - In final step do not calculate rounding error but update matrices directly
            self._xty += upd_xty
            self._xtx += upd_xtx

            # - Weight and bias update by ridge regression
            solution = np.linalg.solve(
                self._xtx + regularize * np.eye(input_size), self._xty
            )
            if train_biases:
                self.weights = solution[:-1, :]
                self.bias = solution[-1, :]
            else:
                self.weights = solution

            # - Remove dat stored during this trainig
            self._xty = None
            self._xtx = None
            self.kahan_comp_xty = None
            self.kahan_comp_xtx = None
            self._training_state = None

    def train_logreg(
        self,
        ts_target: TSContinuous,
        ts_input: TSEvent = None,
        learning_rate: float = 0,
        regularize: float = 0,
        batch_size: Optional[int] = None,
        epochs: int = 1,
        store_states: bool = True,
        verbose: bool = False,
    ):
        """
        train_logreg - Train self with logistic regression over one of possibly many batches.
                       Note that this training method assumes that a sigmoid funciton is applied
                       to the layer output, which is not the case in self.evolve.
        :param ts_target:    TimeSeries - target for current batch
        :param ts_input:     TimeSeries - input to self for current batch
        :learning_rate:     flaot - Factor determining scale of weight increments at each step
        :regularize:       float - regularization parameter
        :batch_size:        int - Number of samples per batch. If None, train with all samples at once
        :epochs:           int - How many times is training repeated
        :store_states:       bool - Include last state from previous training and store state from this
                                   traning. This has the same effect as if data from both trainings
                                   were presented at once.
        :verbose:          bool - Print output about training progress
        """

        # - Discrete time steps for evaluating input and target time series
        num_timesteps = int(np.round(ts_target.duration / self.dt))
        time_base = self._gen_time_trace(ts_target.t_start, num_timesteps)

        # - Discard last sample to avoid counting time points twice
        time_base = time_base[:-1]

        # - Make sure time_base does not exceed ts_target
        time_base = time_base[time_base <= ts_target.t_stop]

        # - Prepare target data
        target = ts_target(time_base)

        # - Make sure no nan is in target, as this causes learning to fail
        assert not np.isnan(
            target
        ).any(), "Layer `{}`: nan values have been found in target (where: {})".format(
            self.name, np.where(np.isnan(target))
        )

        # - Check target dimensions
        if target.ndim == 1 and self.size == 1:
            target = target.reshape(-1, 1)

        assert (
            target.shape[-1] == self.size
        ), "Layer `{}`: Target dimensions ({}) does not match layer size ({})".format(
            self.name, target.shape[-1], self.size
        )

        # - Prepare input data

        # Empty input array with additional dimension for training biases
        inp = np.zeros((np.size(time_base), self.size_in + 1))
        inp[:, -1] = 1

        # - Generate spike trains from ts_input
        if ts_input is None:
            # - Assume zero input
            print(
                "Layer `{}`: No ts_input defined, assuming input to be 0.".format(
                    self.name
                )
            )

        else:
            # - Get data within given time range
            event_times, event_channels = ts_input(
                t_start=time_base[0], t_stop=time_base[-1]
            )

            # - Make sure that input channels do not exceed layer input dimensions
            try:
                assert (
                    np.amax(event_channels) <= self.size_in - 1
                ), "Layer `{}`: Number of input channels exceeds layer input dimensions.".format(
                    self.name
                )
            except ValueError as e:
                # - No events in input data
                if event_channels.size == 0:
                    print("Layer `{}`: No input spikes for training.".format(self.name))
                else:
                    raise e

            # Extract spike data from the input
            spike_raster = (
                ts_input.raster(
                    dt=self.dt,
                    t_start=time_base[0],
                    num_timesteps=time_base.size,
                    channels=np.arange(self.size_in),
                    add_events=self.add_events,
                )
            ).astype(float)

            if store_states:
                try:
                    # - Include last state from previous batch
                    spike_raster[0, :] += self._training_state
                except AttributeError:
                    pass

            # - Define exponential kernel
            kernel = np.exp(-(np.arange(time_base.size - 1) * self.dt) / self.tau_syn)

            # - Apply kernel to spike trains and add filtered trains to input array
            for channel, events in enumerate(spike_raster.T):
                inp[:, channel] = fftconvolve(events, kernel, "full")[: time_base.size]

        # - Prepare batches for training
        if batch_size is None:
            num_batches = 1
            batch_size = num_timesteps
        else:
            num_batches = int(np.ceil(num_timesteps / float(batch_size)))

        sample_order = np.arange(
            num_timesteps
        )  # Indices to choose samples - shuffle for random order

        # - Iterate over epochs
        for ind_epoch in range(epochs):
            # - Iterate over batches and optimize
            for ind_batch in range(num_batches):
                simple_indices = sample_order[
                    ind_batch * batch_size : (ind_batch + 1) * batch_size
                ]
                # - Gradients
                gradients = self._gradients(
                    inp[simple_indices], target[simple_indices], regularize
                )
                self.weights = self.weights - learning_rate * gradients[:-1, :]
                self.bias = self.bias - learning_rate * gradients[-1, :]
            if verbose:
                print(
                    "Layer `{}`: Training epoch {} of {}".format(
                        self.name, ind_epoch + 1, epochs
                    ),
                    end="\r",
                )
            # - Shuffle samples
            np.random.shuffle(sample_order)

        if verbose:
            print("Layer `{}`: Finished trainig.              ".format(self.name))

        if store_states:
            # - Store last state for next batch
            self._training_state = inp[-1, :-1].copy()

    def _gradients(self, inp, target, regularize):
        # - Output with current weights
        linear = inp[:, :-1] @ self.weights + self.bias
        output = sigmoid(linear)
        # - Gradients for weights
        num_samples = inp.shape[0]
        error = output - target
        gradients = (inp.T @ error) / float(num_samples)
        # - Regularization of weights
        if regularize > 0:
            gradients[:-1, :] += regularize / float(self.size_in) * self.weights

        return gradients

    ### --- Properties

    @property
    def input_type(self):
        return TSEvent

    @property
    def tau_syn(self):
        return self._tau_syn

    @tau_syn.setter
    def tau_syn(self, new_tau_syn):
        assert new_tau_syn > 0, "Layer `{}`: tau_syn must be greater than 0.".format(
            self.name
        )
        self._tau_syn = new_tau_syn

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, new_bias):
        self._bias = self._expand_to_net_size(new_bias, "bias", allow_none=False)

    @property
    def state(self):
        return self._state_no_bias + self._bias

    @state.setter
    def state(self, new_state):
        new_state = np.asarray(self._expand_to_net_size(new_state, "state"))
        self._state_no_bias = new_state - self._bias

    @property
    def xtx(self):
        return self._xtx

    @property
    def xty(self):
        return self._xty
