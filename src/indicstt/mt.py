"""Machine translation: IndicTrans2 distilled ONNX INT8 (indic-indic 320M, indic-en / en-indic 200M).

The bundle for a direction loads lazily on first use, so an engine that never translates pays nothing.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

FLORES = {
    "hi": "hin_Deva", "gu": "guj_Gujr", "mr": "mar_Deva", "kn": "kan_Knda", "ml": "mal_Mlym",
    "ta": "tam_Taml", "te": "tel_Telu", "or": "ory_Orya", "bn": "ben_Beng", "en": "eng_Latn",
}


def _indic_processor():
    """Import IndicProcessor without IndicTransToolkit/__init__, which needs transformers<5 and torch."""
    if "IndicTransToolkit" not in sys.modules:
        spec = importlib.util.find_spec("IndicTransToolkit")
        pkg = types.ModuleType("IndicTransToolkit")
        pkg.__path__ = list(spec.submodule_search_locations)
        sys.modules["IndicTransToolkit"] = pkg
    from IndicTransToolkit.processor import IndicProcessor

    return IndicProcessor(inference=True)


class _SharedWeights:
    """Feeds external-data initializers to sessions as memory-mapped arrays, one array per distinct byte range.

    Identical (file, offset, length) ranges map to the same array, so graphs that share weights share memory.
    Pages come from the file mapping (page cache) instead of private heap copies.
    """

    def __init__(self):
        self._maps: dict[Path, np.memmap] = {}
        self._arrays: dict[tuple, object] = {}  # (file, offset, length) -> OrtValue

    def options(self, model_path: Path, threads: int):
        import onnx
        import onnxruntime as ort
        from onnx.helper import tensor_dtype_to_np_dtype

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        if os.environ.get("INDICSTT_MT_PREPACK", "0") != "1":
            # pre-packing would give every session a private copy of the weights in ORT's packed format
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


class IndicTransONNX:
    """One IndicTrans2 ONNX bundle with greedy decoding and KV cache."""

    def __init__(self, path: str | Path, threads: int = 4):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        p = Path(path)
        prov = ["CPUExecutionProvider"]
        self._shared = _SharedWeights()  # keeps memory-mapped weight arrays alive for the sessions' lifetime
        self.enc = ort.InferenceSession(str(p / "encoder_model.onnx"),
                                        self._shared.options(p / "encoder_model.onnx", threads), providers=prov)
        # decoder and decoder_with_past reference the same bytes of decoder_shared.onnx.data under different
        # names; without sharing, ONNX Runtime loads those ~100 MB twice
        self.dec = ort.InferenceSession(str(p / "decoder_model.onnx"),
                                        self._shared.options(p / "decoder_model.onnx", threads), providers=prov)
        self.dec_past = ort.InferenceSession(str(p / "decoder_with_past_model.onnx"),
                                             self._shared.options(p / "decoder_with_past_model.onnx", threads),
                                             providers=prov)
        self.n_layers = (len(self.dec.get_outputs()) - 1) // 4
        self.src_tok = Tokenizer.from_file(str(p / "tokenizer_src.json"))
        self.tgt_tok = Tokenizer.from_file(str(p / "tokenizer_tgt.json"))
        self.meta = json.loads((p / "tokenizer_meta.json").read_text())
        gen = json.loads((p / "generation_config.json").read_text())
        self.start_id = int(gen.get("decoder_start_token_id", 2))
        self.eos_id = int(gen.get("eos_token_id", 2))
        self.ip = _indic_processor()
        self.lock = threading.Lock()  # IndicProcessor keeps per-call placeholder state

    def translate(self, text: str, src: str, tgt: str) -> str:
        with self.lock:
            if hasattr(self.ip, "_placeholder_entity_maps"):
                self.ip._placeholder_entity_maps.queue.clear()
            pre = self.ip.preprocess_batch([text], src_lang=FLORES[src], tgt_lang=FLORES[tgt])[0]
            enc = self.src_tok.encode(pre)
            unk, src_n = self.meta["unk_id"], self.meta["src_dict_size"]
            ids = np.array([[i if i < src_n else unk for i in enc.ids]], dtype=np.int64)
            mask = np.array([enc.attention_mask], dtype=np.int64)
            hidden = self.enc.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": mask})[0]

            out = [self.start_id]
            cur = np.array([[self.start_id]], dtype=np.int64)
            past = None
            for step in range(min(256, 2 * ids.shape[1] + 16)):
                if step == 0:
                    res = self.dec.run(None, {"input_ids": cur, "encoder_hidden_states": hidden,
                                              "encoder_attention_mask": mask})
                else:
                    feed = {"input_ids": cur, "encoder_attention_mask": mask}
                    for i in range(self.n_layers):
                        b = 4 * i
                        feed[f"past_key_values.{i}.decoder.key"] = past[b]
                        feed[f"past_key_values.{i}.decoder.value"] = past[b + 1]
                        feed[f"past_key_values.{i}.encoder.key"] = past[b + 2]
                        feed[f"past_key_values.{i}.encoder.value"] = past[b + 3]
                    res = self.dec_past.run(None, feed)
                past = res[1:] if step == 0 else _merge_past(past, res[1:])
                nxt = int(np.argmax(res[0][0, -1]))
                out.append(nxt)
                if nxt == self.eos_id:
                    break
                cur = np.array([[nxt]], dtype=np.int64)
            tgt_n = self.meta["tgt_dict_size"]
            raw = self.tgt_tok.decode([i if i < tgt_n else unk for i in out], skip_special_tokens=True)
            return self.ip.postprocess_batch([raw], lang=FLORES[tgt])[0]


def _merge_past(prev, new):
    # decoder_with_past returns only updated self-attention KV; encoder KV stay from step 0
    if len(new) == len(prev):
        return new
    merged = list(prev)
    for i in range(len(prev) // 4):
        merged[4 * i], merged[4 * i + 1] = new[2 * i], new[2 * i + 1]
    return merged


class Translator:
    """Routes a (src, tgt) pair to the right IndicTrans2 bundle."""

    def __init__(self, models_dir: str | Path, threads: int = 4):
        self.dir = Path(models_dir)
        self.threads = threads
        self._bundles: dict[str, IndicTransONNX] = {}
        self._load_lock = threading.Lock()

    def _bundle_name(self, src: str, tgt: str) -> str:
        if src == "en":
            return "mt-en-indic"
        if tgt == "en":
            return "mt-indic-en"
        return "mt-indic-indic"

    def bundle(self, name: str) -> IndicTransONNX:
        with self._load_lock:
            if name not in self._bundles:
                self._bundles[name] = IndicTransONNX(self.dir / name, self.threads)
            return self._bundles[name]

    def available(self, src: str, tgt: str) -> bool:
        return (self.dir / self._bundle_name(src, tgt) / "encoder_model.onnx").exists()

    def translate(self, text: str, src: str, tgt: str) -> tuple[str, float]:
        """Returns (translation, milliseconds)."""
        if src == tgt or not text.strip():
            return text, 0.0
        t = time.perf_counter()
        out = self.bundle(self._bundle_name(src, tgt)).translate(text, src, tgt)
        return out, (time.perf_counter() - t) * 1000
