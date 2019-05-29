##########
# virtual_dynapse.py - This module defines a Layer class that simulates a DynapSE
#                      processor. Its purpose is to provide an understanding of
#                      which operations are possible with the hardware. The
#                      implemented neuron model is a simplification of the actual
#                      circuits and therefore only serves as a rough approximation.
#                      Accordingly, hyperparameters such as time constants or
#                      baseweights give an idea on the parameters that can be set
#                      but there is no direct correspondence to the hardware biases.
# TODO: Mention that ways of extending connectivity exist, but complex.
##########

from NetworksPython.layers import Layer


class VirtualDynapse(Layer):
    def __init__(
        self,
        connections,  # - 3D array, with 0th dim corresponding to synapse type, others connections (pos. integers)
        tau_mem,  # - Array of size 16
        tau_mem_alt,
        tau_syn_fe,
        tau_syn_se,
        tau_syn_fi,
        tau_syn_si,
        t_refractory,
        weights_fe,
        weights_se,
        weights_fi,
        weights_si,
        thresholds,
        # syn_thresholds???
        tau_alt,  # - Binary array with size of
        dt,
    ):
        pass

    def _validate_connections(self, connections):
        """
        validate:
            - limited fan in, by summing connections over dim 0 and then dim 1
            - fan-out limit to 3 chips -> how to check? (sum over dim 0 and check rows ->
            np.unique(v_row // 16).size < 4
        """

    def evolve(self, ts_input, duration, num_timesteps, verbose):
        pass


# Functions:
# - Adaptivity?
# - NMDA synapses?
# - BUF_P???
# - Syn-thresholds???
# Limitations
# - Isi time step, event limit and isi limit??
