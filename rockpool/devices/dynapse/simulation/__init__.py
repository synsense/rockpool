"""
Dynap-SE2 Simulation Module

This module implements a neuron model that solves analog VLSI circuit equations to provide a reliable simulated machine.
JAX-backend Dynap-SE2 simulator named `DynapSim` does not simulate the hardware numerically precisely but executes a fast and an approximate simulation.
It uses forward Euler updates to predict the time-dependent dynamics and solves the characteristic circuit transfer functions in time.

This module also provides a surrogate function implementation to support gradient based optimization and a mismatch prototype to support analog device mismatch simulation.

See also:
    The neuron model tutorial provided in :ref:`/devices/DynapSE/neuron-model.ipynb`
"""
from .dynapsim import *
from .surrogate import *
from .mismatch_prototype import *
