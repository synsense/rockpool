###
# exp_synapses_manual.py - Class implementing a spike-to-current layer with exponential synapses
###


# - Imports
from typing import Optional, Union, Tuple, List, Dict
import numpy as np
from scipy.signal import fftconvolve

from rockpool.timeseries import TSContinuous, TSEvent
from rockpool.nn.layers.layer import Layer
from rockpool.nn.modules.timed_module import astimedmodule


# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = ["FFExpSyn"]

# - Absolute tolerance, e.g. for comparing float values
tol_abs = 1e-9


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


## - FFExpSyn - Class: define an exponential synapse layer (spiking input)
@astimedmodule(
    parameters=["weights", "bias", "tau_syn"],
    simulation_parameters=["dt", "noise_std", "add_events"],
)
class FFExpSyn(Layer):
    """Define an exponential synapse layer with spiking inputs and current outputs"""

    ## - Constructor
    def __init__(
        self,
        weights: Union[np.ndarray, int],
        bias: Union[np.ndarray, float] = 0,
        dt: float = 0.0001,
        noise_std: float = 0,
        tau_syn: float = 0.005,
        name: str = "unnamed",
        add_events: bool = True,
    ):
        """
        Construct an exponential synapse layer (spiking inputs, current outputs)

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

    ### --- State evolution

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        Function to evolve the states of this layer given an input

        :param Optional[TSEvent] ts_input:  Input spike train
        :param Optional[float] duration:    Simulation/Evolution time
        :param Optional[int] num_timesteps: Number of evolution time steps
        :param Optional[bool] verbose:      Currently no effect, just for conformity

        :return TSContinuous:               Output currents
        """

        # - Prepare weighted input signal
        __, inp_raster, num_timesteps = self._prepare_input(
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

        # - Filter input spike trains
        filtered = self._filter_data(weighted_input, num_timesteps=time_base.size)

        # - Update time and state
        self._timestep += num_timesteps
        self._state_no_bias = filtered[-1]

        # - Output time series with output data and bias
        return TSContinuous(time_base, filtered + self.bias, name="Receiver current")

    def train(
        self,
        ts_target: TSContinuous,
        ts_input: TSContinuous,
        is_first: bool,
        is_last: bool,
        method: str = "rr",
        **kwargs,
    ):
        """
        Wrapper to standardize training syntax across layers. Use specified training method to train layer for current batch.

        :param TSContinuous ts_target:  Target time series for current batch.
        :param TSContinuous ts_input:   Input to the layer during the current batch.
        :param bool is_first:           Set ``True`` to indicate that this batch is the first in training procedure.
        :param bool is_last:            Set ``True`` to indicate that this batch is the last in training procedure.
        :param str method:              String indicating which training method to choose. Currently only ridge regression ("rr") and logistic regression are supported.
        :param kwargs:                  Will be passed on to corresponding training method.
        """

        raise NotImplementedError(
            "Training is currently not available for this module."
        )
        # - Choose training method
        if method in {
            "rr",
            "ridge",
            "ridge regression",
            "regression",
            "linear regression",
            "linreg",
        }:
            training_method = self.train_rr
        elif method in {"logreg", "logistic", "logistic regression"}:
            training_method = self.train_logreg
        else:
            raise ValueError(
                f"FFExpSyn `{self.name}`: Training method `{method}` is currently not "
                + "supported. Use `rr` for ridge regression or `logreg` for logistic "
                + "regression."
            )
        # - Call training method
        return training_method(
            ts_target, ts_input, is_first=is_first, is_last=is_last, **kwargs
        )

    def _filter_data(self, data: np.ndarray, num_timesteps: int):
        """
        Filter input data y convolving with the synaptic kernel

        :param np.ndarray data:     Input data
        :param int num_timesteps:   The number of time steps to return

        :return np.ndarray:         The filtered data
        """

        if num_timesteps is None:
            num_timesteps = len(data)

        # - Define exponential kernel
        # Make kernel shorter by setting values smaller than ``tol_abs`` to 0
        len_kernel = -self.tau_syn / self.dt * np.log(tol_abs)
        kernel = np.exp(-np.arange(len_kernel) * self.dt / self.tau_syn)

        # - Make sure spikes only have effect on next time step
        kernel = np.r_[0, kernel]

        # - Apply kernel to spike trains
        filtered = fftconvolve(data, kernel.reshape(-1, 1), "full", axes=0)
        filtered = filtered[:num_timesteps]

        return filtered

    def _prepare_training_data(
        self,
        ts_target: TSContinuous,
        ts_input: Optional[TSEvent] = None,
        is_first: bool = True,
        is_last: bool = False,
    ):
        """
        Check and rasterize input and target signals for this batch

        :param TSContinuous ts_target:      Target signal for this batch
        :param Optional[TSEvent] ts_input:  Input signal for this batch. Default: ``None``, no input for this batch
        :param bool is_first:     If ``True``, this is the first batch in training. Default: ``True``, this is the first batch
        :param bool is_last:      If ``True``, this is the last training batch. Default: ``False``, this is not the last batch

        :return (inp, target, time_base)
            inp np.ndarray:         Rasterized input signal [T, M]
            target np.ndarray:      Rasterized target signal [T, O]
            time_base np.ndarray:   Time base for ``inp`` and ``target``
        """
        __, target, time_base = super()._prepare_training_data(
            ts_target, ts_input, is_first, is_last
        )

        # - Prepare input data
        inp = np.zeros((np.size(time_base), self.size_in))

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

            if self._store_states and not is_first:
                try:
                    # - Include last state from previous batch
                    spike_raster[0, :] += self._training_state
                except AttributeError:
                    pass

            # - Filter input spike trains
            inp = self._filter_data(spike_raster, num_timesteps=time_base.size)

        if self._store_states:
            # - Store last state for next batch
            self._training_state = inp[-1, :].copy()

        return inp, target, time_base

    def train_rr(
        self,
        ts_target: TSContinuous,
        ts_input: Union[TSEvent, TSContinuous] = None,
        regularize: float = 0,
        is_first: bool = True,
        is_last: bool = False,
        store_states: bool = True,
        train_biases: bool = True,
        calc_intermediate_results: bool = False,
        return_training_progress: bool = True,
        return_trained_output: bool = False,
        fisher_relabelling: bool = False,
        standardize: bool = False,
        n_prune: int = 0,
    ) -> Union[Dict, None]:
        """
        Train self with ridge regression over one of possibly many batches. Use Kahan summation to reduce rounding errors when adding data to existing matrices from previous batches.

        :param TSContinuous ts_target:                  Target for current batch
        :param Union[TSEvent, TSContinuous] ts_input:   Input to self for current batch
        :param float regularize:                        Regularization parameter for ridge regression
        :param bool is_first:                           ``True`` if current batch is the first in training
        :param bool is_last:                            ``True`` if current batch is the last in training
        :param bool store_states:                       If ``True``, include last state from previous training and store state from this training. This has the same effect as if data from both trainings were presented at once.
        :param bool train_biases:                       If ``True``, train biases as if they were weights Otherwise present biases will be ignored in training and not be changed.
        :param bool calc_intermediate_results:          If ``True``, calculates the intermediate weights not in the final batch
        :param bool return_training_progress:           If ``True``, return dict of current training variables for each batch.
        :param bool standardize:                        If ``True``, train with z-score standardized data, based on means and standard deviations from first batch
        :param n_prune: int                       Number of coefficients to prune. Pruning is only applied if is_last == True.

        :return Union[None, dict]:
            If `return_training_progress`, return dict with current training variables (xtx, xty, kahan_comp_xtx, kahan_comp_xty).
            Weights and biases are returned if `is_last` or if `calc_intermediate_results`.
            If `return_trained_output`, the dict contains the output of evolving with the newly trained weights.
        """

        raise NotImplementedError(
            "Training is currently not available for this module."
        )

        self._store_states = store_states
        tr_data = super().train_rr(
            ts_target=ts_target,
            ts_input=ts_input,
            regularize=regularize,
            is_first=is_first,
            is_last=is_last,
            train_biases=train_biases,
            calc_intermediate_results=calc_intermediate_results,
            return_training_progress=return_training_progress,
            return_trained_output=return_trained_output,
            fisher_relabelling=fisher_relabelling,
            standardize=standardize,
            n_prune=n_prune,
        )

        if store_states and return_training_progress:
            tr_data["training_progress"]["training_state"] = self._training_state

        if return_trained_output or return_training_progress:
            return tr_data

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
        Train self with logistic regression over one of possibly many batches. Note that this training method assumes that a sigmoid function is applied to the layer output, which is not the case in `.evolve`.

        :param TSContinuous ts_target:  Target for current batch
        :param TSEvent ts_input:        Input to self for current batch
        :param float learning_rate:     Factor determining scale of weight increments at each step
        :param float regularize:        Regularization parameter
        :param int batch_size:          Number of samples per batch. If None, train with all samples at once
        :param int epochs:              How many times is training repeated
        :param bool store_states:       Include last state from previous training and store state from this training. This has the same effect as if data from both trainings were presented at once.
        :param bool verbose:            Print output about training progress
        """

        raise NotImplementedError(
            "Training is currently not available for this module."
        )

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

    def _gradients(
        self, inp: np.ndarray, target: np.ndarray, regularize: float
    ) -> np.ndarray:
        """
        Compute gradients for this batch

        :param np.ndarray inp:      Input time series for this batch [T, M]
        :param np.ndarray target:   Target time series for this batch [T, O]
        :param float regularize:    Regularization parameter for weights. Reduces the L1-norm of the weights (weight sum)

        :return np.ndarray:         Gradients for weights
        """
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

    def to_dict(self) -> dict:
        """
        Convert parameters of ``self`` to a dict if they are relevant for reconstructing an identical layer
        """
        # - Basic layer attributes from super class
        config = super().to_dict()
        # - add class-specific attributes
        config["bias"] = self.bias if type(self._bias) is float else self._bias.tolist()
        config["tau_syn"] = (
            self.tau_syn if type(self.tau_syn) is float else self.tau_syn.tolist()
        )
        config["name"] = self.name
        config["add_events"] = self.add_events

        return config

    ### --- Properties

    @property
    def input_type(self):
        """ (`.TSEvent`) Time series class accepted by this layer (`.TSEvent`) """
        return TSEvent

    @property
    def tau_syn(self):
        """ (float) Output synaptic time constants for this layer """
        return self._tau_syn

    @tau_syn.setter
    def tau_syn(self, new_tau_syn):
        assert new_tau_syn > 0, "Layer `{}`: tau_syn must be greater than 0.".format(
            self.name
        )
        self._tau_syn = new_tau_syn

    @property
    def bias(self):
        """ (np.ndarray) Bias currents for the neurons in this layer [N,]"""
        return self._bias

    @bias.setter
    def bias(self, new_bias):
        self._bias = self._expand_to_net_size(new_bias, "bias", allow_none=False)

    @property
    def state(self):
        """ (np.ndarray) Internal neuron state of the neurons in this layer [N,] """
        return self._state_no_bias + self._bias

    @state.setter
    def state(self, new_state):
        new_state = np.asarray(self._expand_to_net_size(new_state, "state"))
        self._state_no_bias = new_state - self._bias

    @property
    def xtx(self):
        """ (np.ndarray) $X^{T}X$ intermediate training value """
        return self._xtx

    @property
    def xty(self):
        """ (np.ndarray) $X^{T}Y$ intermediate training value """
        return self._xty
