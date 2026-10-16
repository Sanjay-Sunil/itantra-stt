"""ONNX Runtime session helpers: memory-mapped, shareable weights and memory settings."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SharedWeights:
    """Feeds external-data initializers to sessions as memory-mapped arrays, one array per distinct byte range.

    Identical (file, offset, length) ranges map to the same array, so graphs that share weights share memory.
    Pages come from the file mapping (page cache) instead of private heap copies.
    """

    def __init__(self):
        self._maps: dict[Path, np.memmap] = {}
        self._arrays: dict[tuple, object] = {}  # (file, offset, length) -> OrtValue

    def options(self, model_path: Path, threads: int, prepack: bool = False, arena: bool = True):
        import onnx
        import onnxruntime as ort
        from onnx.helper import tensor_dtype_to_np_dtype

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = arena
        if not prepack:
            # pre-packing gives every session a private copy of the weights in ORT's packed format
            so.add_session_config_entry("session.disable_prepacking", "1")
        graph = onnx.load(str(model_path), load_external_data=False).graph
        for init in graph.initializer:
            if init.data_location != onnx.TensorProto.EXTERNAL:
                continue
            kv = {e.key: e.value for e in init.external_data}
            f = model_path.parent / kv["location"]
            key = (f, int(kv.get("offset", 0)), int(kv["length"]))
            if key not in self._arrays:
                if f not in self._maps:
                    self._maps[f] = np.memmap(f, dtype=np.uint8, mode="r")
                raw = self._maps[f][key[1] : key[1] + key[2]]
                arr = raw.view(tensor_dtype_to_np_dtype(init.data_type)).reshape(tuple(init.dims))
                self._arrays[key] = ort.OrtValue.ortvalue_from_numpy(arr)
            so.add_initializer(init.name, self._arrays[key])
        return so


def plain_options(threads: int, prepack: bool = True, arena: bool = True):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = arena
    if not prepack:
        so.add_session_config_entry("session.disable_prepacking", "1")
    return so


def session(model_path: str | Path, threads: int, *, prepack: bool = True, arena: bool = True,
            mmap: bool = False, shared: SharedWeights | None = None):
    """InferenceSession; mmap=True feeds external weights as memory-mapped arrays (shared via `shared`)."""
    import onnxruntime as ort

    model_path = Path(model_path)
    if mmap or shared is not None:
        so = (shared or SharedWeights()).options(model_path, threads, prepack=prepack, arena=arena)
    else:
        so = plain_options(threads, prepack, arena)
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
