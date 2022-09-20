"""
Dynap-SE weight quantization package provides easy to use support

Note : Existing modules are reconstructed considering consistency with Xylo support.

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022
"""
from typing import Any, Dict, Optional
import numpy as np


def autoencoder_quantization(
    weights_in: np.ndarray,
    weights_rec: np.ndarray,
    Iw_base: float,
    bits_per_weight: Optional[int] = 4,
) -> Dict[Any, Any]:

    if bits_per_weight > 4:
        raise ValueError("Up-to 4-bits representation supported")

    w_in = None
    w_rec = None
    Iw_0 = None
    Iw_1 = None
    Iw_2 = None
    Iw_3 = None

    return {
        "weights_in": w_in,
        "weights_rec": w_rec,
        "Iw_0": Iw_0,
        "Iw_1": Iw_1,
        "Iw_2": Iw_2,
        "Iw_3": Iw_3,
    }
