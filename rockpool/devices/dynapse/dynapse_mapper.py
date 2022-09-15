"""
Dynap-SE graph graph mapper package

- Create a graph using the :py:meth:`~.graph.GraphModule.as_graph` API
- Call :py:func:`.mapper`

Note : Existing modules are reconstructed considering consistency with Xylo support.


Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022
"""

__all__ = ["mapper", "DRCError", "DRCWarning"]

from typing import Any, Dict, Union

import numpy as np

from rockpool.graph import GraphModuleBase


class DRCError(ValueError):
    pass


class DRCWarning(Warning, DRCError):
    pass


def mapper(
    graph: GraphModuleBase,
    weight_dtype: Union[np.dtype, str] = "float",
) -> Dict[Any]:
    pass
