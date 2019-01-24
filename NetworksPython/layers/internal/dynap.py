###
# dynap.py - Classes implementing layers for simulating on the DynapSE
###

from warnings import warn

# -- Attempt to import required CtxCTL modules
try:
    from .dynap_hw import RecDynapSE

except ModuleNotFoundError:
    # - Let's try to use a spike-based simulation instead
    warn('A connection to the required DynapSE hardware was not found. Will try to use a spiking simulation instead.')
    from .pytorch import RecIAFSpkInTorch as RecDynapSE
