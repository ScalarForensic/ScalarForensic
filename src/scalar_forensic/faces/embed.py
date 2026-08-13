"""Operator-supplied ONNX face embedder, described by a JSON manifest (spec §6.3).

CPU execution provider only in Phase 1; SFN_DEVICE does not apply here.
Threads are pinned to avoid OpenMP oversubscription against torch in the
same process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort

from scalar_forensic.embedder import hash_file

_REQUIRED_FIELDS = (
    "input_name",
    "layout",
    "channel_order",
    "dtype",
    "input_size",
    "mean",
    "scale",
    "output_name",
    "embedding_dim",
)


@dataclass(frozen=True)
class EmbedderManifest:
    input_name: str
    layout: Literal["NCHW", "NHWC"]
    channel_order: Literal["RGB", "BGR"]
    dtype: Literal["float32"]
    input_size: int
    mean: float
    scale: float
    output_name: str
    embedding_dim: int


def load_manifest(model_path: Path) -> EmbedderManifest:
    manifest_path = Path(str(model_path) + ".manifest.json")
    if not manifest_path.exists():
        raise ValueError(f"Embedder manifest not found: {manifest_path}")
    raw = json.loads(manifest_path.read_text())
    missing = [f for f in _REQUIRED_FIELDS if f not in raw]
    if missing:
        raise ValueError(f"Embedder manifest {manifest_path} missing fields: {', '.join(missing)}")
    # Coerce numerics: JSON "128" vs "128.0" must not change normalization_id,
    # which is a hard comparability field — two identical manifests would
    # otherwise refuse each other.
    raw["mean"] = float(raw["mean"])
    raw["scale"] = float(raw["scale"])
    raw["input_size"] = int(raw["input_size"])
    raw["embedding_dim"] = int(raw["embedding_dim"])
    m = EmbedderManifest(**{f: raw[f] for f in _REQUIRED_FIELDS})
    if m.layout not in ("NCHW", "NHWC"):
        raise ValueError(f"manifest layout must be NCHW or NHWC, got {m.layout!r}")
    if m.channel_order not in ("RGB", "BGR"):
        raise ValueError(f"manifest channel_order must be RGB or BGR, got {m.channel_order!r}")
    if m.input_size != 112:
        raise ValueError(f"manifest input_size must be 112 (arcface-112-v1), got {m.input_size}")
    return m


class OnnxFaceEmbedder:
    def __init__(self, model_path: Path) -> None:
        self.manifest = load_manifest(model_path)
        self.model_hash = hash_file(model_path)
        self.manifest_hash = hash_file(Path(str(model_path) + ".manifest.json"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._validate_session()
        # Some embedder exports (SFace among them) declare a fixed batch dim
        # instead of a dynamic one; feeding more crops than declared is an
        # ONNXRuntime INVALID_ARGUMENT error, so embed() must chunk to this.
        in_shape = self._session.get_inputs()[0].shape
        self._max_batch: int | None = in_shape[0] if isinstance(in_shape[0], int) else None
        self.embedding_norms: np.ndarray = np.empty(0, dtype=np.float32)

    @property
    def normalization_id(self) -> str:
        return f"affine-{self.manifest.mean}-{self.manifest.scale}"

    def _validate_session(self) -> None:
        m = self.manifest
        inputs = {i.name: i for i in self._session.get_inputs()}
        outputs = {o.name: o for o in self._session.get_outputs()}
        if m.input_name not in inputs:
            raise ValueError(f"model has no input {m.input_name!r}; found {list(inputs)}")
        if m.output_name not in outputs:
            raise ValueError(f"model has no output {m.output_name!r}; found {list(outputs)}")
        out_shape = outputs[m.output_name].shape
        # ONNX dynamic dims come back as str symbols — only validate concrete ints.
        if isinstance(out_shape[-1], int) and out_shape[-1] != m.embedding_dim:
            raise ValueError(
                f"manifest embedding_dim={m.embedding_dim} but model emits {out_shape[-1]}"
            )

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        m = self.manifest
        batch = np.stack(crops).astype(np.float32)
        if m.channel_order == "BGR":
            batch = batch[..., ::-1]
        batch = (batch - m.mean) / m.scale
        if m.layout == "NCHW":
            batch = batch.transpose(0, 3, 1, 2)
        batch = np.ascontiguousarray(batch)
        if self._max_batch is not None and len(batch) > self._max_batch:
            raw = np.vstack(
                [
                    self._session.run(
                        [m.output_name], {m.input_name: batch[i : i + self._max_batch]}
                    )[0]
                    for i in range(0, len(batch), self._max_batch)
                ]
            )
        else:
            (raw,) = self._session.run([m.output_name], {m.input_name: batch})
        norms = np.linalg.norm(raw, axis=1)
        self.embedding_norms = norms.astype(np.float32)
        return (raw / np.clip(norms[:, None], 1e-12, None)).astype(np.float32)
