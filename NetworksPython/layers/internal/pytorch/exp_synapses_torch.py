###
# exp_synapses_torch.py - Like exp_synapses_manual.py but with torch for computations.
###


# - Imports
import json
from warnings import warn
from typing import Union, Optional
import numpy as np
from scipy.signal import fftconvolve
import torch

from ....timeseries import TSContinuous, TSEvent
from ..exp_synapses_manual import FFExpSyn
from ...layer import ArrayLike, RefProperty


# - Configure exports
__all__ = ["FFExpSyn"]

# - Absolute tolerance, e.g. for comparing float values
tol_abs = 1e-9
# - Default maximum numbers of time steps for a single evolution batch
nDefaultMaxNumTimeSteps = 5000


def sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))


## - FFExpSynTorch - Class: define an exponential synapse layer (spiking input, pytorch as backend)
class FFExpSynTorch(FFExpSyn):
    """ FFExpSynTorch - Class: define an exponential synapse layer (spiking input, pytorch as backend)
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
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        FFExpSynTorch - Construct an exponential synapse layer (spiking input, pytorch as backend)

        :param weights:             np.array MxN weight matrix
                                int Size of layer -> creates one-to-one conversion layer
        :param dt:             float Time step for state evolution
        :param noise_std:       float Std. dev. of noise added to this layer. Default: 0

        :param tau_syn:         float Output synaptic time constants. Default: 5ms
        :param synapse_eq:      Brian2.Equations set of synapse equations for receiver. Default: exponential
        :param integrator_name:   str Integrator to use for simulation. Default: 'exact'

        :param name:         str Name for the layer. Default: 'unnamed'

        :add_events:            bool  If during evolution multiple input events arrive during one
                                      time step for a channel, count their actual number instead of
                                      just counting them as one.

        :nMaxNumTimeSteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Set device to cuda if available and determine how tensors should be instantiated
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.tensors = torch.cuda
        else:
            self.device = torch.device("cpu")
            print(
                "Layer `{}`: Using CPU because CUDA is not available.".format(name)
            )
            self.tensors = torch


        # - Bypass property setter to avoid unnecessary convolution kernel update
        assert (
            type(nMaxNumTimeSteps) == int and nMaxNumTimeSteps > 0.0
        ), "Layer `{}`: nMaxNumTimeSteps ({step}) must be an integer greater than 0.".format(
            name, step=nMaxNumTimeSteps
        )
        self._nMaxNumTimeSteps = nMaxNumTimeSteps

        # - Call super constructor
        super().__init__(
            weights=weights,
            bias=bias,
            dt=dt,
            noise_std=noise_std,
            tau_syn=tau_syn,
            name=name,
            add_events=add_events,
        )

    ### --- State evolution

    # @profile
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

        # - Prepare input signal
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

        with torch.no_grad():
            # - Tensor for collecting output spike raster
            output = torch.FloatTensor(num_timesteps + 1, self.size).fill_(0)

            # - Iterate over batches and run evolution
            iCurrentIndex = 1
            for mfCurrentInput, nCurrNumTS in self._batch_data(
                weighted_input, num_timesteps, self.nMaxNumTimeSteps
            ):
                output[
                    iCurrentIndex : iCurrentIndex + nCurrNumTS
                ] = self._single_batch_evolution(
                    mfCurrentInput,  # torch.from_numpy(mfCurrentInput).float().to(self.device),
                    nCurrNumTS,
                    verbose,
                )
                iCurrentIndex += nCurrNumTS

        # - Output time series with output data and bias
        return TSContinuous(
            time_base,
            (output + self._bias.cpu()).numpy(),
            name="Filtered spikes",
        )

    # @profile
    def _batch_data(
        self, inp: np.ndarray, num_timesteps: int, nMaxNumTimeSteps: int = None
    ) -> (np.ndarray, int):
        """_batch_data: Generator that returns the data in batches"""
        # - Handle None for nMaxNumTimeSteps
        nMaxNumTimeSteps = (
            num_timesteps if nMaxNumTimeSteps is None else nMaxNumTimeSteps
        )
        nStart = 0
        while nStart < num_timesteps:
            # - Endpoint of current batch
            nEnd = min(nStart + nMaxNumTimeSteps, num_timesteps)
            # - Data for current batch
            mfCurrentInput = (
                torch.from_numpy(inp[nStart:nEnd]).float().to(self.device)
            )
            yield mfCurrentInput, nEnd - nStart
            # - Update nStart
            nStart = nEnd

    # @profile
    def _single_batch_evolution(
        self, weighted_input: np.ndarray, num_timesteps: int, verbose: bool = False
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param weighted_input: np.ndarray   Weighted input
        :param num_timesteps:   int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        with torch.no_grad():

            # Add current state to input
            weighted_input[0, :] += self._state_no_bias.clone() * np.exp(
                -self.dt / self.tau_syn
            )

            # - Reshape input for convolution
            weighted_input = weighted_input.t().reshape(1, self.size, -1)

            # - Filter synaptic currents
            filtered = (
                self._convSynapses(weighted_input)[0].detach().t()[:num_timesteps]
            )

            # - Store current state and update internal time
            self._state_no_bias = filtered[-1].clone()
            self._timestep += num_timesteps

        return filtered

    def train_rr(
        self,
        ts_target: TSContinuous,
        ts_input: TSEvent = None,
        regularize: float = 0,
        is_first: bool = True,
        is_last: bool = False,
        store_states: bool = True,
        train_biases: bool = True,
        bIntermediateResults: bool = False
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
        :param bIntermediateResults: bool - If True, calculates the intermediate weights not in the final batch
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

        with torch.no_grad():
            # - Move target data to GPU
            target = torch.from_numpy(target).float().to(self.device)

            # - Prepare input data

            # Empty input array with additional dimension for training biases
            input_size = self.size_in + int(train_biases)
            inp = self.tensors.FloatTensor(time_base.size, input_size).fill_(0)
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
                    print(
                        "Layer `{}`: No input spikes for training.".format(self.name)
                    )
                else:
                    raise e

            with torch.no_grad():
                # Extract spike data from the input variable and bring to GPU
                spike_raster = (
                    torch.from_numpy(
                        ts_input.raster(
                            dt=self.dt,
                            t_start=time_base[0],
                            num_timesteps=time_base.size,
                            channels=np.arange(self.size_in),
                            add_events=self.add_events,
                        ).astype(float)
                    )
                    .float()
                    .to(self.device)
                )

                if store_states and not is_first:
                    try:
                        # - Include last state from previous batch
                        spike_raster[0, :] += self._training_state
                    except AttributeError:
                        pass

                # - Reshape input for convolution
                spike_raster = spike_raster.t().reshape(1, self.size_in, -1)

                # - Filter synaptic currents and store in input tensor
                if train_biases:
                    inp[:, :-1] = (
                        self._convSynapsesTraining(spike_raster)[0]
                        .detach()
                        .t()[: time_base.size]
                    )
                else:
                    inp[:, :] = (
                        self._convSynapsesTraining(spike_raster)[0]
                            .detach()
                            .t()[: time_base.size]
                    )


        with torch.no_grad():
            # - For first batch, initialize summands
            if is_first:
                # Matrices to be updated for each batch
                self._xty = self.tensors.FloatTensor(input_size, self.size).fill_(0)
                self._xtx = self.tensors.FloatTensor(input_size, input_size).fill_(0)
                # Corresponding Kahan compensations
                self._kahan_comp_xty = self._xty.clone()
                self._mfKahanCompXTX = self._xtx.clone()

            # - New data to be added, including compensation from last batch
            #   (Matrix summation always runs over time)
            upd_xty = torch.mm(inp.t(), target) - self._kahan_comp_xty
            upd_xtx = torch.mm(inp.t(), inp) - self._mfKahanCompXTX

            if not is_last:
                # - Update matrices with new data
                new_xty = self._xty + upd_xty
                new_xtx = self._xtx + upd_xtx
                # - Calculate rounding error for compensation in next batch
                self._kahan_comp_xty = (new_xty - self._xty) - upd_xty
                self._mfKahanCompXTX = (new_xtx - self._xtx) - upd_xtx
                # - Store updated matrices
                self._xty = new_xty
                self._xtx = new_xtx

                if store_states:
                    # - Store last state for next batch
                    if train_biases:
                        self._training_state = inp[-1, :-1].clone()
                    else:
                        self._training_state = inp[-1, :].clone()

                if bIntermediateResults:
                    a = self._xtx + regularize * torch.eye(self.size_in + 1).to(
                        self.device
                    )
                    solution = torch.mm(a.inverse(), self._xty).cpu().numpy()
                    if train_biases:
                        self.weights = solution[:-1, :]
                        self.bias = solution[-1, :]
                    else:
                        self.weights = solution
            else:
                # - In final step do not calculate rounding error but update matrices directly
                self._xty += upd_xty
                self._xtx += upd_xtx

                # - Weight and bias update by ridge regression
                if train_biases:
                    a = self._xtx + regularize * torch.eye(self.size_in + 1).to(
                        self.device
                    )
                else:
                    a = self._xtx + regularize * torch.eye(self.size_in).to(
                        self.device
                    )

                solution = torch.mm(a.inverse(), self._xty).cpu().numpy()
                if train_biases:
                    self.weights = solution[:-1, :]
                    self.bias = solution[-1, :]
                else:
                    self.weights = solution

                # - Remove data stored during this trainig
                self._xty = None
                self._xtx = None
                self._kahan_comp_xty = None
                self._mfKahanCompXTX = None
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
                       Use pytorch as backend
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

        if not torch.cuda.is_available():
            warn("Layer `{}`: CUDA not available. Will use cpu".format(self.name))
            self.train_logreg(
                ts_target,
                ts_input,
                learning_rate,
                regularize,
                batch_size,
                epochs,
                store_states,
                verbose,
            )
            return

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
                    print(
                        "Layer `{}`: No input spikes for training.".format(self.name)
                    )
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
            kernel = np.exp(
                -(np.arange(time_base.size - 1) * self.dt) / self.tau_syn
            )

            # - Apply kernel to spike trains and add filtered trains to input array
            for channel, events in enumerate(spike_raster.T):
                inp[:, channel] = fftconvolve(events, kernel, "full")[
                    : time_base.size
                ]

        # - Move data to cuda
        ctTarget = torch.from_numpy(target).float().to("cuda")
        ctInput = torch.from_numpy(inp).float().to("cuda")
        ctWeights = torch.from_numpy(self.weights).float().to("cuda")
        ctBiases = torch.from_numpy(self.bias).float().to("cuda")

        # - Prepare batches for training
        if batch_size is None:
            num_batches = 1
            batch_size = num_timesteps
        else:
            num_batches = int(np.ceil(num_timesteps / float(batch_size)))

        ctSampleOrder = torch.arange(
            num_timesteps
        )  # Indices to choose samples - shuffle for random order

        # - Iterate over epochs
        for ind_epoch in range(epochs):
            # - Iterate over batches and optimize
            for ind_batch in range(num_batches):
                ctSampleIndices = ctSampleOrder[
                    ind_batch * batch_size : (ind_batch + 1) * batch_size
                ]
                # - Gradients
                ctGradients = self._gradients(
                    ctWeights,
                    ctBiases,
                    ctInput[ctSampleIndices],
                    ctTarget[ctSampleIndices],
                    regularize,
                )
                ctWeights -= learning_rate * ctGradients[:-1, :]
                ctBiases -= learning_rate * ctGradients[-1, :]
            if verbose:
                print(
                    "Layer `{}`: Training epoch {} of {}".format(
                        self.name, ind_epoch + 1, epochs
                    ),
                    end="\r",
                )
            # - Shuffle samples
            ctSampleOrder = torch.randperm(num_timesteps)

        if verbose:
            print("Layer `{}`: Finished trainig.              ".format(self.name))

        if store_states:
            # - Store last state for next batch
            self._training_state = ctInput[-1, :-1].cpu().numpy()

    def _gradients(self, ctWeights, ctBiases, ctInput, ctTarget, regularize):
        # - Output with current weights
        ctLinear = torch.mm(ctInput[:, :-1], ctWeights) + ctBiases
        ctOutput = sigmoid(ctLinear)
        # - Gradients for weights
        num_samples = ctInput.size()[0]
        ctError = ctOutput - ctTarget
        ctGradients = torch.mm(ctInput.t(), ctError) / float(num_samples)
        # - Regularization of weights
        if regularize > 0:
            ctGradients[:-1, :] += regularize / float(self.size_in) * ctWeights

        return ctGradients

    def _update_kernels(self):
        """Generate kernels for filtering input spikes during evolution and training"""
        nKernelSize = min(
            50
            * int(
                self._tau_syn / self.dt
            ),  # - Values smaller than ca. 1e-21 are neglected
            self._nMaxNumTimeSteps
            + 1,  # Kernel does not need to be larger than batch duration
        )
        times = (
            torch.arange(nKernelSize).to(self.device).reshape(1, -1).float() * self.dt
        )

        # - Kernel for filtering recurrent spikes
        mfInputKernels = torch.exp(
            -times / self.tensors.FloatTensor(self.size, 1).fill_(self._tau_syn)
        )
        # - Reverse on time axis and reshape to match convention of pytorch
        mfInputKernels = mfInputKernels.flip(1).reshape(self.size, 1, nKernelSize)
        # - Object for applying convolution
        self._convSynapses = torch.nn.Conv1d(
            self.size,
            self.size,
            nKernelSize,
            padding=nKernelSize - 1,
            groups=self.size,
            bias=False,
        ).to(self.device)
        self._convSynapses.weight.data = mfInputKernels

        # - Kernel for filtering recurrent spikes (uses unweighted input and therefore has different dimensions)
        mfInputKernelsTraining = self.tensors.FloatTensor(
            self.size_in, nKernelSize
        ).fill_(0)
        mfInputKernelsTraining[:, 1:] = torch.exp(
            -times[:, :-1]
            / self.tensors.FloatTensor(self.size_in, 1).fill_(self._tau_syn)
        )
        # - Reverse on time axis and reshape to match convention of pytorch
        mfInputKernelsTraining = mfInputKernelsTraining.flip(1).reshape(
            self.size_in, 1, nKernelSize
        )
        # - Object for applying convolution
        self._convSynapsesTraining = torch.nn.Conv1d(
            self.size_in,
            self.size_in,
            nKernelSize,
            padding=nKernelSize - 1,
            groups=self.size_in,
            bias=False,
        ).to(self.device)
        self._convSynapsesTraining.weight.data = mfInputKernelsTraining

        print("Layer `{}`: Filter kernels have been updated.".format(self.name))

    ### --- Properties

    @property
    def tau_syn(self):
        return self._tau_syn

    @tau_syn.setter
    def tau_syn(self, new_tau_syn, bNoKernelUpdate=False):
        assert new_tau_syn > 0, "Layer `{}`: tau_syn must be greater than 0.".format(
            self.name
        )
        self._tau_syn = new_tau_syn
        self._update_kernels()

    @RefProperty
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, new_bias):
        new_bias = self._expand_to_net_size(new_bias, "bias", allow_none=False)
        self._bias = torch.from_numpy(new_bias).float().to(self.device)

    @RefProperty
    def xtx(self):
        return self._xtx

    @RefProperty
    def xty(self):
        return self._xty

    @property
    def state(self):
        warn(
            "Layer `{}`: Changing values of returned object by item assignment will not have effect on layer's state".format(
                self.name
            )
        )
        return (self._state_no_bias + self._bias).cpu().numpy()

    @state.setter
    def state(self, new_state):
        new_state = np.asarray(self._expand_to_net_size(new_state, "state"))
        self._state_no_bias = (
            torch.from_numpy(new_state).float().to(self.device) - self._bias
        )

    @property
    def nMaxNumTimeSteps(self):
        return self._nMaxNumTimeSteps

    @nMaxNumTimeSteps.setter
    def nMaxNumTimeSteps(self, nNewMax):
        assert (
            type(nNewMax) == int and nNewMax > 0.0
        ), "Layer `{}`: nMaxNumTimeSteps must be an integer greater than 0.".format(
            self.name
        )
        self._nMaxNumTimeSteps = nNewMax
        self._update_kernels()

    def to_dict(self):

        config = {}
        config["weights"] = self.weights.tolist()
        config["bias"] = self._bias if type(self._bias) is float else self._bias.tolist()
        config["dt"] = self.dt
        config["noise_std"] = self.noise_std
        config["tauS"] = self.tau_syn if type(self.tau_syn) is float else self.tau_syn.tolist()
        config["name"] = self.name
        config["add_events"] = self.add_events
        config["nMaxNumTimeSteps"] = self.nMaxNumTimeSteps
        config["class_name"] = "FFExpSynTorch"

        return config

    def save(self, config, filename):
        with open(filename, "w") as f:
            json.dump(config, f)

    @staticmethod
    def load_from_dict(config):
        return FFExpSynTorch(
            weights = config["weights"],
            bias = config["bias"],
            dt = config["dt"],
            noise_std = config["noise_std"],
            tau_syn = config["tauS"],
            name = config["name"],
            add_events = config["add_events"],
            nMaxNumTimeSteps = config["nMaxNumTimeSteps"],
        )
