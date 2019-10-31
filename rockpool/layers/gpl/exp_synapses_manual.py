###
# exp_synapses_manual.py - Class implementing a spike-to-current layer with exponential synapses
###


# - Imports
from typing import Optional, Union, Tuple, List, Dict
import numpy as np
from scipy.signal import fftconvolve

from ...timeseries import TSContinuous, TSEvent
from ..layer import Layer


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
        weights: Union[np.ndarray, int],
        bias: Union[np.ndarray, float] = 0,
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
        train - Wrapper to standardize training syntax across layers. Use
                specified training method to train layer for current batch.
        :param ts_target: Target time series for current batch.
        :param ts_input:  Input to the layer during the current batch.
        :param is_first:  Set `True` to indicate that this batch is the first in training procedure.
        :param is_last:   Set `True` to indicate that this batch is the last in training procedure.
        :param method:    String indicating which training method to choose.
                          Currently only ridge regression ("rr") and logistic
                          regression are supported.
        kwargs will be passed on to corresponding training method.
        """
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

    def train_rr(
        self,
        ts_target: TSContinuous,
        ts_input: TSEvent = None,
        regularize: float = 0,
        is_first: bool = True,
        is_last: bool = False,
        store_states: bool = True,
        train_biases: bool = True,
        calc_intermediate_results: bool = False,
        return_training_progress: bool = True,
        return_trained_output: bool = False,
        fisher_relabelling: bool = False,
    ) -> Union[Dict, None]:
        """
        train_rr - Train self with ridge regression over one of possibly
                   many batches. Use Kahan summation to reduce rounding
                   errors when adding data to existing matrices from
                   previous batches.
        :param ts_target:        TimeSeries - target for current batch
        :param ts_input:         TimeSeries - input to self for current batch
        :regularize:             float - regularization for ridge regression
        :is_first:               bool - True if current batch is the first in training
        :is_last:                bool - True if current batch is the last in training
        :store_states:           bool - Include last state from previous training and store state from this
                                       traning. This has the same effect as if data from both trainings
                                       were presented at once.
        :param train_biases:     bool - If True, train biases as if they were weights
                                        Otherwise present biases will be ignored in
                                        training and not be changed.
        :param calc_intermediate_results: bool - If True, calculates the intermediate weights not in the final batch
        :param return_training_progress:  bool - If True, return dict of current training
                                                 variables for each batch.
        :return:
            If `return_training_progress`, return dict with current trainig variables
            (xtx, xty, kahan_comp_xtx, kahan_comp_xty).
            Weights and biases are returned if `is_last` or if `calc_intermediate_results`.
            If `return_trained_output`, the dict contains the output of evolveing with
            the newly trained weights.
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

        if store_states:
            # - Store last state for next batch
            if train_biases:
                self._training_state = inp[-1, :-1].copy()
            else:
                self._training_state = inp[-1, :].copy()

        self._curr_tr_params = dict(
            return_training_progress=return_training_progress,
            store_states=store_states,
            train_biases=train_biases,
            calc_intermediate_results=calc_intermediate_results,
            return_trained_output=return_trained_output,
            regularize=regularize,
        )

        if fisher_relabelling:
            return self._train_rr_fisher(inp, target, is_first, is_last, time_base)
        else:
            return self._train_rr_standard(inp, target, is_first, is_last, time_base)

    def _train_rr_standard(self, inp, target, is_first, is_last, time_base):
        input_size = inp.shape[1]

        # - For first batch, initialize summands
        if is_first:
            # Matrices to be updated for each batch
            self._xty = np.zeros((input_size, self.size))  # inp.T (dot) target
            self._xtx = np.zeros((input_size, input_size))  # inp.T (dot) inp
            # Corresponding Kahan compensations
            self._kahan_comp_xty = np.zeros_like(self._xty)
            self._kahan_comp_xtx = np.zeros_like(self._xtx)

        if self._curr_tr_params["return_training_progress"]:
            current_trainig_progress = dict()
            if self._curr_tr_params["store_states"]:
                current_trainig_progress["training_state"] = self._training_state

        new_data = self._batch_update(
            inp=inp,
            target=target,
            xty_old=self._xty,
            xtx_old=self._xtx,
            kahan_comp_xty_old=self._kahan_comp_xty,
            kahan_comp_xtx_old=self._kahan_comp_xtx,
            input_size=input_size,
            is_last=is_last,
        )

        if not is_last:
            self._xty = new_data["xty"]
            self._xtx = new_data["xtx"]
            self._kahan_comp_xty = new_data["kahan_comp_xty"]
            self._kahan_comp_xtx = new_data["kahan_comp_xtx"]

        if is_last or self._curr_tr_params["calc_intermediate_results"]:
            # - Update layer weights
            assert new_data["weights"].shape == (self.size_in, self.size)
            self.weights = new_data["weights"]
            if self._curr_tr_params["train_biases"]:
                self.bias = new_data["bias"]

        if is_last:
            # - Remove data stored during this trainig epoch
            self._xty = None
            self._xtx = None
            self._kahan_comp_xty = None
            self._kahan_comp_xtx = None

        if (
            self._curr_tr_params["return_trained_output"]
            or self._curr_tr_params["return_training_progress"]
        ):
            return_data = dict()
            if self._curr_tr_params["return_trained_output"]:
                if self._curr_tr_params["train_biases"]:
                    inp_nobias = inp[:, :-1]
                else:
                    inp_nobias = inp
                output_samples = inp_nobias @ new_data["weights"] + new_data["bias"]
                return_data["output"] = TSContinuous(time_base, output_samples)
            if self._curr_tr_params["return_training_progress"]:
                current_trainig_progress.update(new_data["curr_tr_prog"])
                return_data["current_trainig_progress"] = current_trainig_progress

        return return_data

    def _train_rr_fisher(self, inp, target, is_first, is_last, time_base):
        input_size = inp.shape[1]
        num_timesteps = time_base.size

        # - Relabel target based on number of occurences of corresponding data points
        bool_tgt = target.astype(bool)
        nums_true = np.sum(bool_tgt, axis=0)
        nums_false = num_timesteps - nums_true
        labels_true = num_timesteps / nums_true
        labels_false = num_timesteps / nums_false
        target = target.astype(float)
        for i_tgt, (tgt_vec_bool, lbl_t, lbl_f) in enumerate(
            zip(bool_tgt.T, labels_true, labels_false)
        ):
            target[tgt_vec_bool, i_tgt] = lbl_t
            target[tgt_vec_bool == False, i_tgt] = lbl_f

        # - For first batch, initialize summands
        if is_first:
            self._xty = {}  # inp.T (dot) target
            self._xtx = {}  # inp.T (dot) inp
            self._kahan_comp_xty = {}  # Kahan compensation for xty
            self._kahan_comp_xtx = {}  # Kahan compensation for xtx

            for i_unit in range(self.size):
                self._xty[i_unit] = np.zeros((input_size, 1))
                self._kahan_comp_xty[i_unit] = np.zeros((input_size, 1))
                self._xtx[i_unit] = np.zeros((input_size, input_size))
                self._kahan_comp_xtx[i_unit] = np.zeros((input_size, input_size))

        if self._curr_tr_params["return_trained_output"]:
            output_samples = np.zeros((target.shape[0], target.shape[1]))

        if self._curr_tr_params["return_training_progress"]:
            current_trainig_progress = dict()
            if self._curr_tr_params["store_states"]:
                current_trainig_progress["training_state"] = self._training_state

        for i_unit in range(self.size):
            inp_unit = inp.copy()
            inp_unit[bool_tgt[:, i_unit] == False, :] *= -1
            tgt_unit = target[:, i_unit].reshape(num_timesteps, 1)

            new_data = self._batch_update(
                inp=inp_unit,
                target=tgt_unit,
                xty_old=self._xty[i_unit],
                xtx_old=self._xtx[i_unit],
                kahan_comp_xty_old=self._kahan_comp_xty[i_unit],
                kahan_comp_xtx_old=self._kahan_comp_xtx[i_unit],
                input_size=input_size,
                is_last=is_last,
            )
            if not is_last:
                self._xty[i_unit] = new_data["xty"]
                self._xtx[i_unit] = new_data["xtx"]
                self._kahan_comp_xty[i_unit] = new_data["kahan_comp_xty"]
                self._kahan_comp_xtx[i_unit] = new_data["kahan_comp_xtx"]

            if is_last or self._curr_tr_params["calc_intermediate_results"]:
                # - Update layer weights
                assert new_data["weights"].shape == (self.size_in, 1)
                self.weights[:, i_unit] = new_data["weights"].flatten()
                if self._curr_tr_params["train_biases"]:
                    self.bias[i_unit] = new_data["bias"]

            if self._curr_tr_params["return_trained_output"]:
                if self._curr_tr_params["train_biases"]:
                    inp_nobias = inp[:, :-1]
                else:
                    inp_nobias = inp
                output_unit = inp_nobias @ new_data["weights"] + new_data["bias"]
                output_samples[:, i_unit] = output_unit.flatten()

            if self._curr_tr_params["return_training_progress"]:
                current_trainig_progress[i_unit] = new_data["curr_tr_prog"]

        if is_last:
            # - Remove data stored during this trainig epoch
            self._xty = None
            self._xtx = None
            self._kahan_comp_xty = None
            self._kahan_comp_xtx = None

        if (
            self._curr_tr_params["return_trained_output"]
            or self._curr_tr_params["return_training_progress"]
        ):
            return_data = dict()
            if self._curr_tr_params["return_trained_output"]:
                return_data["output"] = TSContinuous(time_base, output_samples)
            if self._curr_tr_params["return_training_progress"]:
                return_data["current_trainig_progress"] = current_trainig_progress

        return return_data

    def _batch_update(
        self,
        inp,
        target,
        xty_old,
        xtx_old,
        kahan_comp_xty_old,
        kahan_comp_xtx_old,
        input_size,
        is_last,
    ):
        # - New data to be added, including compensation from last batch
        #   (Matrix summation always runs over time)
        upd_xty = inp.T @ target - kahan_comp_xty_old
        upd_xtx = inp.T @ inp - kahan_comp_xtx_old

        # - Collect new data in dict
        new_data = dict()
        # - Update matrices with new data
        new_data["xty"] = xty_old + upd_xty
        new_data["xtx"] = xtx_old + upd_xtx

        if self._curr_tr_params["return_training_progress"]:
            new_data["curr_tr_prog"] = dict(xty=new_data["xty"], xtx=new_data["xtx"])

        if not is_last:
            # - Calculate rounding error for compensation in next batch
            kahan_comp_xty_new = (new_data["xty"] - xty_old) - upd_xty
            kahan_comp_xtx_new = (new_data["xtx"] - xtx_old) - upd_xtx
            new_data["kahan_comp_xty"] = kahan_comp_xty_new
            new_data["kahan_comp_xtx"] = kahan_comp_xtx_new

            if self._curr_tr_params["return_training_progress"]:
                new_data["curr_tr_prog"]["kahan_comp_xty"] = kahan_comp_xty_new
                new_data["curr_tr_prog"]["kahan_comp_xtx"] = kahan_comp_xtx_new

            if (
                self._curr_tr_params["calc_intermediate_results"]
                or self._curr_tr_params["return_trained_output"]
            ):
                regularization = self._curr_tr_params["regularize"]
                reg_data = new_data["xtx"] + regularization * np.eye(input_size)
                solution = np.linalg.solve(reg_data, new_data["xty"])
                if self._curr_tr_params["train_biases"]:
                    new_data["weights"] = solution[:-1, :]
                    new_data["bias"] = solution[-1, :]
                else:
                    new_data["weights"] = solution
                if self._curr_tr_params["return_training_progress"]:
                    new_data["curr_tr_prog"]["weights"] = new_data["weights"]
                    if self._curr_tr_params["train_biases"]:
                        new_data["curr_tr_prog"]["bias"] = new_data["bias"]

        else:
            # - Weight and bias update by ridge regression
            regularization = self._curr_tr_params["regularize"]
            reg_data = new_data["xtx"] + regularization * np.eye(input_size)
            solution = np.linalg.solve(reg_data, new_data["xty"])
            if self._curr_tr_params["train_biases"]:
                new_data["weights"] = solution[:-1, :]
                new_data["bias"] = solution[-1, :]
            else:
                new_data["weights"] = solution
            if self._curr_tr_params["return_training_progress"]:
                new_data["curr_tr_prog"]["weights"] = new_data["weights"]
                if self._curr_tr_params["train_biases"]:
                    new_data["curr_tr_prog"]["bias"] = new_data["bias"]

        return new_data

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

    def to_dict(self) -> dict:
        """
        to_dict - Convert parameters of `self` to a dict if they are relevant for
                  reconstructing an identical layer.
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
