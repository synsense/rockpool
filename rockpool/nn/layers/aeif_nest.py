"""
Adaptive-exponential integrate-and-fire neurons with NEST backend
"""

import multiprocessing
from warnings import warn
from typing import Union, List, Dict

import numpy as np

from .iaf_nest import (
    RecIAFSpkInNestV1,
    _BaseNestProcessSpkInRec,
    s2ms,
    V2mV,
    mV2V,
    F2mF,
    COMMAND_GET,
    COMMAND_SET,
)
from rockpool.utilities.property_arrays import SetterArray, ImmutableArray
from rockpool.nn.modules.timed_module import astimedmodule


# - RecAEIFSpkInNest- Class: Spiking recurrent layer with spiking in- and outputs
class RecAEIFSpkInNestV1(RecIAFSpkInNestV1):
    """Spiking recurrent layer with spiking in- and outputs, with a NEST backend"""

    class NestProcess(_BaseNestProcessSpkInRec):
        """Class for running NEST in its own process"""

        def __init__(
            self,
            request_q,
            result_q,
            weights_in: np.ndarray,
            weights_rec: np.ndarray,
            delay_in: Union[float, np.ndarray],
            delay_rec: Union[float, np.ndarray],
            bias: Union[float, np.ndarray],
            dt: float,
            tau_syn_exc: Union[float, np.ndarray],
            tau_syn_inh: Union[float, np.ndarray],
            conductance: Union[float, np.ndarray],
            capacity: Union[float, np.ndarray],
            v_thresh: Union[float, np.ndarray],
            v_peak: Union[float, np.ndarray],
            v_reset: Union[float, np.ndarray],
            v_rest: Union[float, np.ndarray],
            refractory,
            a: Union[float, np.ndarray],
            b: Union[float, np.ndarray],
            delta_t: Union[float, np.ndarray],
            tau_w: Union[float, np.ndarray],
            record: bool = False,
            num_cores: int = 1,
        ):
            """initializes the process"""
            super().__init__(
                request_q=request_q,
                result_q=result_q,
                weights_in=weights_in,
                weights_rec=weights_rec,
                delay_in=delay_in,
                delay_rec=delay_rec,
                bias=bias,
                dt=dt,
                tau_syn_exc=tau_syn_exc,
                tau_syn_inh=tau_syn_inh,
                capacity=capacity,
                v_thresh=v_thresh,
                v_reset=v_reset,
                v_rest=v_rest,
                refractory=refractory,
                record=record,
                num_cores=num_cores,
                model="aeif_psc_exp",
            )

            # - Record weights and layer-specific parameters
            self.v_peak = V2mV(v_peak)
            self.a = a
            self.b = b
            self.delta_t = V2mV(delta_t)
            self.tau_w = s2ms(tau_w)
            self.conductance = conductance

        ######### DEFINE IPC COMMANDS ######

        def reset(self):
            """
            reset_all - IPC command which resets time and state
            """
            self.nest_module.ResetNetwork()
            self.nest_module.SetKernelStatus({"time": 0.0})
            # - Manually reset state parameters
            for name in ("I_syn_ex", "I_syn_in", "w"):
                self.set_param(name, 0.0)
            self.set_param("V_m", self.v_rest)

        def generate_nest_params_list(self) -> List[Dict[str, np.ndarray]]:
            """init_nest_params - Initialize nest neuron parameters and return as list"""

            params = super().generate_nest_params_list()
            for n in range(self.size):
                params[n]["g_L"] = self.conductance[n]
                params[n]["V_peak"] = self.v_peak[n]
                params[n]["a"] = self.a[n]
                params[n]["b"] = self.b[n]
                params[n]["Delta_T"] = self.delta_t[n]
                params[n]["tau_w"] = self.tau_w[n]

            return params

    # - Default difference between v_peak and v_thresh when v_peak not set and
    #   delta_t != 0
    _v_peak_offset = 0.01

    ## - Constructor
    def __init__(
        self,
        weights_in: np.ndarray,
        weights_rec: np.ndarray,
        delay_in: Union[float, np.ndarray] = 0.0001,
        delay_rec: Union[float, np.ndarray] = 0.0001,
        bias: Union[float, np.ndarray] = 0.0,
        dt: float = 0.0001,
        tau_mem: Union[float, np.ndarray] = 0.02,
        tau_syn: Union[np.ndarray, float, None] = 0.05,
        tau_syn_exc: Union[float, np.ndarray, None] = None,
        tau_syn_inh: Union[float, np.ndarray, None] = None,
        v_thresh: Union[float, np.ndarray] = -0.055,
        v_reset: Union[float, np.ndarray] = -0.065,
        v_rest: Union[float, np.ndarray] = -0.065,
        capacity: Union[float, np.ndarray, None] = None,
        conductance: Union[float, np.ndarray, None] = 1.0,
        refractory: Union[float, np.ndarray] = 0.001,
        subthresh_adapt: Union[float, np.ndarray] = 4.0,
        spike_adapt: Union[float, np.ndarray] = 80.5,
        delta_t: Union[float, np.ndarray] = 0.002,
        tau_adapt: Union[float, np.ndarray] = 0.144,
        name: str = "unnamed",
        record: bool = False,
        num_cores: int = 1,
    ):
        """
        Construct a spiking recurrent layer with AEIF neurons, with a NEST back-end in- and outputs are spiking events

        :param weights_in:          np.array MxN input weight matrix.
        :param weights_rec:         np.array NxN recurrent weight matrix.
        :param bias:                np.array Nx1 bias current vector in nA. Default: 0

        :param dt:                  float Time-step in seconds. Default: 0.0001

        :param tau_mem:             np.array Nx1 vector of neuron time constants in seconds. Default: 0.05
        :param tau_syn:             np.array Nx1 vector of synapse time constants in seconds. Used
                                    Used instead of `tau_syn_exc` or `tau_syn_inh` if they are
                                    None. Default: 0.02
        :param tau_syn_exc:         np.array Nx1 vector of excitatory synapse time constants in seconds.
                                    If `None`, use `tau_syn`. Default: `None`
        :param tau_syn_inh:         np.array Nx1 vector of inhibitory synapse time constants in seconds.
                                    If `None`, use `tau_syn`. Default: `None`

        :param v_thresh:            np.array Nx1 vector of neuron thresholds ("point of no return") in Volt. Default: -0.055
        :param v_reset:             np.array Nx1 vector of neuron reset potential in Volt. Default: -0.065V
        :param v_rest:              np.array Nx1 vector of neuron resting potential in Volt. Default: -0.065V

        :param capacity:            np.array Nx1 vector of neuron membrance capacity in nF.
                                    Will be set to `tau_mem` * `conductance` if `None`. Default: `None`.
        :param conductance:         np.array Nx1 vector of neuron leak conductance in nS. Default: 1.0
        :param refractory:          float Refractory period after each spike in seconds. Default: 0.001

        :param subthresh_adapt:     float or np.ndarray scaling for subthreshold adaptation in nS. Default: 4.
        :param spike_adapt:         float or np.ndarray additive value for spike triggered adaptation in nA. Default: 80.5
        :param delta_t:             float or np.ndarray scaling for exponential part of the activation function in Volt.
                                    Default: 0.002
        :param tau_adapt:           float or np.ndarray time constant for adaptation relaxation in seconds. Default: 0.144


        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions
        """

        # - Determine layer size and name to run `_expand_to_net_size` method
        self._size_in, self._size = np.atleast_2d(weights_in).shape
        self.name = name

        # - Handle tau_mem, conductance and capacity
        error_many_nones: str = (
            self.start_print
            + "Of the parameters `tau_mem`, "
            + "`conductance`, and `capacity` only one can be `None`."
            + "You may set `conductance` to 1 (nS)."
        )  # Exception to be raised when more than one of the three parameters is `None`

        if tau_mem is not None:
            tau_mem = self._expand_to_net_size(
                tau_mem, "tau_mem", allow_none=False
            ).astype(float)
            if conductance is not None:
                self._conductance = self._expand_to_net_size(
                    conductance, "conductance", allow_none=False
                ).astype(float)
                self._capacity = self._conductance * tau_mem
                if capacity is not None:
                    warn(
                        self.start_print
                        + "The parameters `tau_mem`, "
                        + "`conductance`, and `capacity` are not independent (`tau_mem` = "
                        + "`capacity` / `conductance`). Will overwrite the value given for "
                        + "`capacity` to `tau_mem` / `conductance`."
                    )
            elif capacity is not None:
                self._capacity = self._expand_to_net_size(
                    capacity, "capacity", allow_none=False
                ).astype(float)
                self._conductance = self._capacity / tau_mem
            else:
                raise ValueError(error_many_nones)
        else:
            if conductance is not None:
                self._conductance = self._expand_to_net_size(
                    conductance, "conductance", allow_none=False
                ).astype(float)
                if capacity is not None:
                    self._capacity = self._expand_to_net_size(
                        capacity, "capacity", allow_none=False
                    ).astype(float)
                    tau_mem = self._capacity / self._conductance
                else:
                    raise ValueError(error_many_nones)
            else:
                raise ValueError(error_many_nones)

        # - Prepare parameters that are specific to this class
        self._subthresh_adapt = self._expand_to_net_size(
            subthresh_adapt, "subthresh_adapt", allow_none=False
        ).astype(float)
        self._spike_adapt = self._expand_to_net_size(
            spike_adapt, "spike_adapt", allow_none=False
        ).astype(float)
        self._tau_adapt = self._expand_to_net_size(
            tau_adapt, "tau_adapt", allow_none=False
        ).astype(float)
        delta_t = self._expand_to_net_size(delta_t, "delta_t", allow_none=False)
        self._delta_t = delta_t.astype(float)

        # - Determine v_thresh to determine v_peak (otherwise done by super().__init__)
        v_thresh = self._expand_to_net_size(v_thresh, "v_thresh", allow_none=False)
        self._v_peak = v_thresh.copy().astype(float)
        self._v_peak[self._delta_t != 0] += self._v_peak_offset
        # else:
        #     v_peak = self._expand_to_net_size(v_peak, "v_peak", allow_none=False)
        #     self._v_peak = v_peak.astype(float)

        # - Call super constructor
        super().__init__(
            weights_in=weights_in,
            weights_rec=weights_rec,
            delay_in=delay_in,
            delay_rec=delay_rec,
            bias=bias,
            dt=dt,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            tau_syn_exc=tau_syn_exc,
            tau_syn_inh=tau_syn_inh,
            v_thresh=v_thresh,
            v_reset=v_reset,
            v_rest=v_rest,
            capacity=self._capacity,
            refractory=refractory,
            name=name,
            record=record,
            num_cores=num_cores,
        )

    def _setup_nest(self):
        """_setup_nest - Set up and start a nest process"""
        self.request_q = multiprocessing.Queue()
        self.result_q = multiprocessing.Queue()

        self._nest_process = self.NestProcess(
            self.request_q,
            self.result_q,
            weights_in=self._weights_in,
            weights_rec=self._weights_rec,
            delay_in=self._delay_in,
            delay_rec=self._delay_rec,
            bias=self._bias,
            dt=self._dt,
            tau_syn_exc=self._tau_syn_exc,
            tau_syn_inh=self._tau_syn_inh,
            capacity=self._capacity,
            conductance=self._conductance,
            v_thresh=self._v_thresh,
            v_peak=self._v_peak,
            v_reset=self._v_reset,
            v_rest=self._v_rest,
            refractory=self._refractory,
            record=self._record,
            num_cores=self._num_cores,
            a=self._subthresh_adapt,
            b=self._spike_adapt,
            delta_t=self._delta_t,
            tau_w=self._tau_adapt,
        )
        self._nest_process.start()

    def to_dict(self) -> dict:
        """
        Convert parameters of this layer to a dict if they are relevant for reconstructing an identical layer

        :return Dict:   A dictionary that can be used to reconstruct the layer
        """
        config = super().to_dict()
        config["subthresh_adapt"] = self._subthresh_adapt.tolist()
        config["spike_adapt"] = self._spike_adapt.tolist()
        config["delta_t"] = self._delta_t.tolist()
        config["tau_adapt"] = self._tau_adapt.tolist()
        config["class_name"] = "RecAEIFSpkInNest"

        return config

    ### --- Properties

    @property
    def adapt(self):
        self.request_q.put([COMMAND_GET, "w"])
        return mV2V(np.array(self.result_q.get()))

    @property
    def tau_mem(self):
        return SetterArray(
            (self._capacity / self._conductance).copy(), owner=self, name="tau_mem"
        )

    @tau_mem.setter
    def tau_mem(self, new_tau):
        new_tau = self._expand_to_net_size(new_tau, "tau_mem", allow_none=False)
        self.capacity = self.conductance * new_tau.astype(float)
        print(
            f"RecAEIFSpkInNest `{self.name}`: `tau_mem` has been updated by modifying "
            + "`capacity` (`capacity` = `tau_mem` * `conductance`)."
        )
        self._tau_mem = new_tau
        self.request_q.put([COMMAND_SET, "C_m", F2mF(self._conductance * new_tau)])

    @property
    def conductance(self):
        return SetterArray(self._conductance, owner=self, name="conductance")

    @conductance.setter
    def conductance(self, new_conductance):
        new_conductance = self._expand_to_net_size(
            new_conductance, "conductance", allow_none=False
        ).astype(float)
        self._conductance = new_conductance
        self.request_q.put([COMMAND_SET, "g_L", new_conductance])

    @property
    def subthresh_adapt(self):
        return SetterArray(self._subthresh_adapt, owner=self, name="subthresh_adapt")

    @subthresh_adapt.setter
    def subthresh_adapt(self, new_a):
        new_a = self._expand_to_net_size(new_a, "subthresh_adapt", allow_none=False)
        new_a = new_a.astype(float)
        self._subthresh_adapt = new_a
        self.request_q.put([COMMAND_SET, "a", new_a])

    @property
    def spike_adapt(self):
        return SetterArray(self._spike_adapt, owner=self, name="spike_adapt")

    @spike_adapt.setter
    def spike_adapt(self, new_b):
        new_b = self._expand_to_net_size(new_b, "spike_adapt", allow_none=False)
        new_b = new_b.astype(float)
        self._spike_adapt = new_b
        self.request_q.put([COMMAND_SET, "b", new_b])

    @property
    def delta_t(self):
        return SetterArray(self._delta_t, owner=self, name="delta_t")

    @delta_t.setter
    def delta_t(self, new_delta_t):
        new_delta_t = self._expand_to_net_size(new_delta_t, "delta_t", allow_none=False)
        new_delta_t = new_delta_t.astype(float)
        self._delta_t = new_delta_t
        self.request_q.put([COMMAND_SET, "Delta_T", V2mV(new_delta_t)])

    @RecIAFSpkInNestV1.v_thresh.setter
    def v_thresh(self, new_v_thresh):
        new_v_thresh = self._expand_to_net_size(
            new_v_thresh, "v_thresh", allow_none=False
        )
        self._v_thresh = new_v_thresh.astype(float)
        self._v_peak = self._v_thresh.copy()
        self._v_peak[self._delta_t != 0] += self._v_peak_offset
        # print("Vth:", self._v_thresh)
        # print("Vpeak:", self._v_peak)
        self.request_q.put([COMMAND_SET, "V_peak", V2mV(self._v_peak)])
        self.request_q.put([COMMAND_SET, "V_th", V2mV(self._v_thresh)])

    @property
    def v_peak(self):
        return ImmutableArray(self._v_peak, name=self.start_print + "`v_peak`")

    @property
    def tau_adapt(self):
        return SetterArray(self._tau_adapt, owner=self, name="tau_adapt")

    @tau_adapt.setter
    def tau_adapt(self, new_tau):
        new_tau = self._expand_to_net_size(new_tau, "tau_adapt", allow_none=False)
        new_tau = new_tau.astype(float)
        self._tau_adapt = new_tau
        self.request_q.put([COMMAND_SET, "tau_w", s2ms(new_tau)])


@astimedmodule(
    parameters=[
        "weights_in",
        "weights_rec",
        "delay_in",
        "delay_rec",
        "bias",
        "tau_mem",
        "tau_syn_exc",
        "tau_syn_inh",
        "capacity",
        "v_thresh",
        "v_reset",
        "v_rest",
        "refractory",
        "conductance",
        "subthresh_adapt",
        "spike_adapt",
        "delta_t",
        "tau_adapt",
    ],
    simulation_parameters=["dt", "record", "num_cores"],
    states=["Vmem"],
)
class RecAEIFSpkInNest(RecAEIFSpkInNestV1):
    pass
