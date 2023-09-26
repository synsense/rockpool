"""
" Provide an accelerated version of event encoding for simulating the AFE on Xylo A2 (SYNS61201)
"""

import numpy as np

from typing import Union, Tuple

from logging import debug, info

__all__ = [
    "_encode_spikes",
    # "_encode_spikes_cpp",
    "_encode_spikes_jax",
    "_encode_spikes_python",
]

# - Try to use C++ version for speedup
# try:
#     # check if the C++ library is available
#     from xylo_a2_spike_generation import _encode_spikes as cpp_encode_spikes

#     # register C++ version
#     __CPP_VERSION__ = True

# except ModuleNotFoundError as e:
#     # try to install C++ module
#     try:
#         info(
#             f"C++ based spike generation module is not installed: {e}. Trying to install this module ..."
#         )

#         # find the library in the path
#         from pathlib import Path
#         import os
#         import subprocess

#         current_dir = Path(__file__).parent.resolve()
#         cpp_library_name = "cpp_xylo_a2_spike_generation"

#         if cpp_library_name not in os.listdir(current_dir):
#             raise ModuleNotFoundError(
#                 f"C++ library: {cpp_library_name} was not found in the current directory!"
#             )

#         # folder containing the C++ code
#         cpp_library_dir = os.path.join(current_dir, cpp_library_name, "src-cpp")

#         # install it using command line
#         command = f'pip install -e "{cpp_library_dir}"'
#         output = subprocess.run(command, shell=True, capture_output=True, text=True)

#         if output.returncode != 0:
#             # command did not run: pip installation did not work
#             info(f"pip installation was not successful: {output.stdout}")
#             raise ModuleNotFoundError(output.stderr)

#         # module was installed so import it again
#         from xylo_a2_spike_generation import _encode_spikes as cpp_encode_spikes

#         # register C++ version
#         __CPP_VERSION__ = True

#     except ModuleNotFoundError as e:
#         info(
#             f"C++ spike generation library was not successful: {e}. Falling back on Jax or Python version for spike generation."
#         )

#         __CPP_VERSION__ = False

#         def cpp_encode_spikes(*args, **kwargs):
#             raise NotImplementedError("CPP version of spike encoding not available.")


# def _encode_spikes_cpp(
#     initial_state: np.ndarray,
#     dt: float,
#     data: np.ndarray,
#     v2i_gain: float,
#     c_iaf: float,
#     leakage: float,
#     thr_up: float,
#     vcc: float,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     C++ version: Encode a signal as events using an LIF neuron membrane with negligible leakage (very close to IAF neuron).

#     Args:
#         initial_state (np.ndarray): Initial state of the LIF neurons.
#         dt (float): Time-step in seconds.
#         data (np.ndarray): Array ``(T,N)`` containing data to convert to events.
#         v2i_gain (float): the gain by which the voltage at the output of rectifier is converted to a current for integration followed by spike generation.
#         c_iaf (float): Membrane capacitance.
#         leakage (float): Leakage factor per time step modelled as a conductance in parallel with capacitor.
#         thr_up (float): Firing threshold voltage.
#         vcc (float): voltage supply on the chip (this is the maximum value of the integrator voltage).

#     Returns:
#         np.ndarray: Raster of output events ``(T,N)``, where ``True`` indicates a spike
#     """
#     # embed the list results in numpy array: this is needed because:
#     # (i)   data is passed as a list to C++ and we would like each row to correspond to one channel
#     # (ii)  result returned from C++ is of the format N xT and we need to convert it into T x N format for compatibility.

#     spikes, final_state = cpp_encode_spikes(
#         initial_state=initial_state,
#         dt=dt,
#         data=data.T,
#         v2i_gain=v2i_gain,
#         c_iaf=c_iaf,
#         leakage=leakage,
#         thr_up=thr_up,
#         vcc=vcc,
#     )

#     return np.asarray(spikes).T, np.asarray(final_state)


# - Try to define a Jax-accelerated version
try:
    import jax
    import jax.numpy as jnp

    __JAX_VERSION__ = True

    info(
        "Jax was detected. Spike generation can be performed by the JIT compiled version."
    )

    @jax.jit
    def _encode_spikes_jax(
        initial_state: np.ndarray,
        dt: float,
        data: np.ndarray,
        v2i_gain: float,
        c_iaf: float,
        leakage: float,
        thr_up: float,
        vcc: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Jax version: Encode a signal as events using an LIF neuron membrane with negligible leakage (very close to IAF neuron).

        Args:
            initial_state (np.ndarray): Initial state of the LIF neurons.
            dt (float): Time-step in seconds.
            data (np.ndarray): Array ``(T,N)`` containing data to convert to events.
            v2i_gain (float): the gain by which the voltage at the output of rectifier is converted to a current for integration followed by spike generation.
            c_iaf (float): Membrane capacitance.
            leakage (float): Leakage factor per time step modelled as a conductance in parallel with capacitor.
            thr_up (float): Firing threshold voltage.
            vcc (float): voltage supply on the chip (this is the maximum value of the integrator voltage).

        Returns:
            np.ndarray: Raster of output events ``(T,N)``, where ``True`` indicates a spike
        """
        # convert the output of the rectifier to a current to be integrated by the capacitor.
        data = data * v2i_gain

        def forward(cdc, data_rec):
            # leakage current when the cpacitor has a voltage of cdc
            lk = leakage * cdc

            # how much charge is depleted from the capacitor during `dt`
            dq_lk = lk * dt

            # how much charge is added to the capacitor because of input data in `dt`
            dq_data = dt * data_rec

            # variation in capacitor voltage dur to data + leakage
            dv = (dq_data - dq_lk) / (c_iaf)

            # - Accumulate membrane voltage, clip to the range [0, VCC]
            cdc += dv

            # truncate the values to the range [0, max_output]
            cdc = jnp.where(cdc < 0.0, 0.0, cdc)
            cdc = jnp.where(cdc > vcc, vcc, cdc)

            spikes = cdc >= thr_up

            return cdc * (1 - spikes), spikes

        # - Evolve over the data array
        final_state, data_up = jax.lax.scan(forward, initial_state, data)

        return jnp.array(data_up), final_state

except Exception as e:
    __JAX_VERSION__ = False

    info(
        f"Jax not available: {e}. Falling back on C++ or Python version for spike generation."
    )

    def _encode_spikes_jax(*args, **kwargs):
        raise NotImplementedError("Jax version of spike encoding not available.")


def _encode_spikes_python(
    initial_state: np.ndarray,
    dt: float,
    data: np.ndarray,
    v2i_gain: float,
    c_iaf: float,
    leakage: float,
    thr_up: float,
    vcc: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Python version: Encode a signal as events using an LIF neuron membrane with negligible leakage (very close to IAF neuron).

    Args:
        initial_state (np.ndarray): Initial state of the LIF neurons.
        dt (float): Time-step in seconds.
        data (np.ndarray): Array ``(T,N)`` containing data to convert to events.
        v2i_gain (float): the gain by which the voltage at the output of rectifier is converted to a current for integration followed by spike generation.
        c_iaf (float): Membrane capacitance.
        leakage (float): Leakage factor per time step modelled as a conductance in parallel with capacitor.
        thr_up (float): Firing threshold voltage.
        vcc (float): voltage supply on the chip (this is the maximum value of the integrator voltage).

    Returns:
        np.ndarray: Raster of output events ``(T,N)``, where ``True`` indicates a spike
    """
    # convert the output of the rectifier to a current to be integrated by the capacitor.
    data = np.copy(data) * v2i_gain

    cdc = initial_state
    spike_list = []

    for i in range(len(data)):
        # leakage current when the cpacitor has a voltage of cdc
        lk = leakage * cdc

        # how much charge is depleted from the capacitor during `dt`
        dq_lk = lk * dt

        # how much charge is added to the capacitor because of input data in `dt`
        dq_data = dt * data[i]

        # variation in capacitor voltage dur to data + leakage
        dv = (dq_data - dq_lk) / (c_iaf)

        # - Accumulate membrane voltage, clip to the range [0, VCC]
        cdc += dv

        cdc[cdc < 0.0] = 0.0
        cdc[cdc > vcc] = vcc

        spikes = cdc >= thr_up

        spike_list.append(spikes)

        cdc = cdc * (1 - spikes)

    spike_list = np.asarray(spike_list)

    return spike_list, cdc


# - In debug mode deactivate accelerated versions
__DEBUG_MODE__ = False

if __DEBUG_MODE__:
    __CPP_VERSION__ = False
    __JAX_VERSION__ = False


if __JAX_VERSION__:
    # Jax version is active: use jax since it is slightly faster than CPP if all dependencies are ok!
    # apply simple embedding in Python
    info(
        f"__JAX_VERSION__: {__JAX_VERSION__}: Using Jax-JIT version of spike encoding."
    )
    _encode_spikes = _encode_spikes_jax

# elif __CPP_VERSION__:
#     # C++ version is active: apply simple embedding in Python
#     info(f"__CPP_VERSION__: {__CPP_VERSION__}: Using C++ version of spike encoding.")
#     _encode_spikes = _encode_spikes_cpp


else:
    # use the Python version
    info(f"No C++ version, no Jax version: Using Python native spike encoding.")
    _encode_spikes = _encode_spikes_python
