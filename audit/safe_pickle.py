"""Load a downloaded pickle without running code: only containers, numpy arrays and tensors are allowed.
Tensors inside a plain pickle are stored as torch.save bytes; those are read with torch's weights_only loader."""
import io
import pickle

import torch

ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
           ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
           ("numpy", "ndarray"), ("numpy", "dtype"), ("collections", "OrderedDict"),
           ("torch._utils", "_rebuild_tensor_v2")}


def _tensor_bytes(b):
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return _tensor_bytes
        if (module, name) not in ALLOWED and not (module == "torch" and name.endswith("Storage")):
            raise pickle.UnpicklingError(f"refusing {module}.{name}")
        return super().find_class(module, name)


def load(path):
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()
