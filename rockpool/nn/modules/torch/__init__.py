"""
Modules using Torch as a backend
"""

# import sys
# from importlib.abc import Loader, MetaPathFinder, FileLoader
# from rockpool.utilities.backend_management import AbortImport
# import importlib

#
# class MyLoader(FileLoader):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.loader = args[0]
#
#     @staticmethod
#     def create_module(*_, **__):
#         return None
#
#     @staticmethod
#     def exec_module(*args, **kwargs):
#         try:
#             importlib.util.exec_module(*args, **kwargs)
#         except AbortImport:
#             pass
#
#
# class MyFinder(MetaPathFinder):
#     def find_spec(self, fullname, path, target=None):
#         print("--- find ---")
#         print(fullname)
#         print(path)
#         if fullname.startswith("rockpool."):
#             return importlib.util.spec_from_loader(fullname, MyLoader)
#         else:
#             return None
#
#
# print("--- init ---")
# __path__ = []
# sys.meta_path.append(MyFinder())


from .torch_module import *
from .rate_torch import *
from .lowpass import *
from .lif_torch import *
from .lif_bitshift_torch import *
from .lif_neuron_torch import *
from .exp_syn_torch import *
from .updown_torch import *
from .linear_torch import *
