"""Session-scoped face detection for *uploaded* query images (spec §8, Phase 1b).

Deliberately separate from ``faces/indexing.py``: that module builds PointStructs
and writes chip files, and the query path must not be one argument away from
persisting anything.  Every pipeline *step* here is imported from
``scalar_forensic.faces.*`` — only the ~70-line orchestration differs, and it is
the part that must provably not persist.

Two invariants this module exists to hold:

* Nothing reaches Qdrant or ``SFN_FACE_STORE_DIR``.  The review crop lives in
  memory on the ``QueryFace`` and dies with the session.
* A face that fails the embedding gate keeps ``vector=None``.  That is the same
  structural exclusion the index path gets from writing a vectorless point —
  a probe with no vector cannot be sent to a kNN, which is why the search
  endpoint can refuse it outright instead of trusting a flag.

The gate order and the surfaces the gates measure are kept identical to
``FacePipeline.process_image``: sharpness and exposure are measured on the
*undilated* source crop, not on the dilated review crop, so the same face
gates the same way whether it arrives by index or by upload.
"""

from __future__ import annotations

import importlib.metadata
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from scalar_forensic.config import Settings
from scalar_forensic.faces.align import ALIGNMENT_VERSION, align_face
from scalar_forensic.faces.chips import dilated_clamped_bbox
from scalar_forensic.faces.decode import load_for_detection
from scalar_forensic.faces.detect import YuNetDetector
from scalar_forensic.faces.embed import OnnxFaceEmbedder
from scalar_forensic.faces.provenance import PipelineConfig
from scalar_forensic.faces.quality import post_align_gate, pre_align_gate, review_gate
from scalar_forensic.web.session import QueryFace

# Matches faces/chips.py — the in-memory review crop is the same artefact the
# store would have written, just never written.
_REVIEW_QUALITY = 95

# Process-lifetime model caches.  The web process loads YuNet and the ONNX
# embedder lazily on the first query-face call; both are stateless across calls
# apart from OnnxFaceEmbedder.embedding_norms, which this module does not read.
_detector_cache: dict[str, YuNetDetector] = {}
_embedder_cache: dict[str, OnnxFaceEmbedder] = {}


def _sfn_version() -> str:
    try:
        return importlib.metadata.version("scalar-forensic")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - dev checkout
        return "unknown"


def _detector(settings: Settings) -> YuNetDetector:
    key = f"{settings.face_detector_model}:{settings.face_detect_max_size}"
    if key not in _detector_cache:
        _detector_cache[key] = YuNetDetector(
            settings.face_detector_model, settings.face_detect_max_size
        )
    return _detector_cache[key]


def _embedder(settings: Settings) -> OnnxFaceEmbedder:
    key = str(settings.face_embedder_model)
    if key not in _embedder_cache:
        _embedder_cache[key] = OnnxFaceEmbedder(settings.face_embedder_model)
    return _embedder_cache[key]


def pipeline_config(settings: Settings, detector, embedder) -> PipelineConfig:
    """The provenance record for a query-time pass.

    Built the same way as ``FacePipeline.from_settings`` (faces/indexing.py) so
    ``FaceStore.check_compat`` compares like with like — the query embedding is
    only meaningful against the collection if it was produced under the same
    hard fields.
    """
    return PipelineConfig(
        detector_id=detector.detector_id,
        detector_model_hash=detector.model_hash,
        detector_score_threshold=detector._score_threshold,
        detect_max_size=settings.face_detect_max_size,
        embedder_model_name=Path(settings.face_embedder_model).name,
        embedder_model_hash=embedder.model_hash,
        manifest_hash=embedder.manifest_hash,
        embedder_dim=embedder.manifest.embedding_dim,
        alignment_version=ALIGNMENT_VERSION,
        normalization_id=embedder.normalization_id,
        min_conf=settings.face_min_conf,
        min_size=settings.face_min_size,
        min_sharpness=settings.face_min_sharpness,
        max_clipped=settings.face_max_clipped,
        max_pose=settings.face_max_pose,
        crop_dilation=settings.face_crop_dilation,
        review_min_conf=settings.face_review_min_conf,
        review_min_size=settings.face_review_min_size,
        sfn_version=_sfn_version(),
        cv2_version=cv2.__version__,
        ort_version=ort.__version__,
    )


@dataclass
class QueryFaceResult:
    faces: list[QueryFace]
    n_detected: int
    n_searchable: int
    n_review_only: int
    rejected: dict[str, int]
    truncated: bool
    cfg: PipelineConfig


def query_embedder_block(cfg: PipelineConfig) -> dict[str, object]:
    """The embedder identity block every face endpoint returns.

    The dimension is read from the manifest via the config, never hardcoded:
    the spec says 512-d, the adopted SFace model is 128-d, and only the
    manifest is authoritative.
    """
    payload = cfg.to_payload()
    keys = (
        "embedder_model_name",
        "embedder_model_hash",
        "manifest_hash",
        "embedder_dim",
        "normalization_id",
        "alignment_version",
    )
    return {k: payload.get(k) for k in keys}


def detect_query_faces(data: bytes, settings: Settings, *, max_faces: int) -> QueryFaceResult:
    """Detect, gate, align and embed the faces in one uploaded image.

    Writes nothing.  Returns the faces in detection order; ``index`` is the
    position in the returned list and is what the client passes back as a probe
    selector — vectors never leave this process.
    """
    img = load_for_detection(data)
    detector = _detector(settings)
    embedder = _embedder(settings)
    cfg = pipeline_config(settings, detector, embedder)

    dets = detector.detect(img)
    n_detected = len(dets)

    faces: list[QueryFace] = []
    rejected: dict[str, int] = {}
    to_embed: list[tuple[int, np.ndarray]] = []
    truncated = False
    h, w = img.shape[:2]

    for det in dets:
        # Truncate on *retained* faces, not on raw detections: the cap exists to
        # bound what the examiner has to choose between, and rejected
        # detections are never offered.
        if len(faces) >= max_faces:
            truncated = True
            break

        rev = review_gate(
            det,
            min_conf=settings.face_review_min_conf,
            min_size=settings.face_review_min_size,
        )
        if not rev.passed:
            reason = rev.reason or "unknown"
            rejected[reason] = rejected.get(reason, 0) + 1
            continue

        # The review crop must exist, or the observation has no reason to be
        # (mirrors indexing.py).
        x0, y0, cw, ch = dilated_clamped_bbox(det.bbox, settings.face_crop_dilation, w, h)
        if cw <= 0 or ch <= 0:
            rejected["size"] = rejected.get("size", 0) + 1
            continue
        review_crop = img[y0 : y0 + ch, x0 : x0 + cw]
        if review_crop.size == 0:
            rejected["size"] = rejected.get("size", 0) + 1
            continue
        buf = io.BytesIO()
        Image.fromarray(review_crop).save(buf, format="JPEG", quality=_REVIEW_QUALITY)

        pre = pre_align_gate(
            det,
            min_conf=settings.face_min_conf,
            min_size=settings.face_min_size,
            max_pose=settings.face_max_pose,
        )
        # None, not 0.0, for anything not yet measured: a 0.0 pose would read as
        # "perfectly frontal" and a 0.0 sharpness as "completely blurred"
        # (spec §6.2).  The post-align subscores are filled in only if that gate
        # actually runs.
        quality: dict[str, float | None] = {
            "confidence": pre.subscores.get("confidence"),
            "size": pre.subscores.get("size"),
            "pose": pre.subscores.get("pose"),
            "sharpness": None,
            "exposure": None,
        }

        # Created review-only and promoted to "embedded" only after a vector
        # actually exists.  The default is the safe one: a bug in what follows
        # leaves a face out of search rather than in it with no vector.
        face = QueryFace(
            index=len(faces),
            bbox=tuple(float(v) for v in det.bbox),
            landmarks=[[float(a), float(b)] for a, b in det.landmarks],
            det_conf=float(det.confidence),
            detect_scale=float(det.detect_scale),
            quality=quality,
            embedding_status="review_only",
            embedding_exclusion_reason=pre.reason,
            vector=None,
            review_jpeg=buf.getvalue(),
        )
        faces.append(face)
        if not pre.passed:
            continue

        aligned = align_face(img, det.landmarks)
        # Sharpness/exposure on the native-resolution *undilated* crop, exactly
        # as indexing.py does: the dilated review crop carries background the
        # index path never measured, so measuring it here would gate the same
        # face differently depending on how it entered the tool.  Stops clamped
        # to >= 0 as well as starts (see indexing.py for the negative-index
        # trap this avoids).
        bx, by, bw, bh = (int(v) for v in det.bbox)
        crop = img[max(0, by) : max(0, by + bh), max(0, bx) : max(0, bx + bw)]
        if crop.size == 0:
            face.embedding_exclusion_reason = "size"
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        post = post_align_gate(
            gray,
            min_sharpness=settings.face_min_sharpness,
            max_clipped_frac=settings.face_max_clipped,
        )
        face.quality["sharpness"] = post.subscores.get("sharpness")
        face.quality["exposure"] = post.subscores.get("exposure")
        if not post.passed:
            face.embedding_exclusion_reason = post.reason
            continue
        to_embed.append((face.index, aligned))

    if to_embed:
        vecs = embedder.embed([a for _, a in to_embed])
        for (idx, _), vec in zip(to_embed, vecs, strict=True):
            faces[idx].vector = [float(v) for v in vec]
            faces[idx].embedding_status = "embedded"
            faces[idx].embedding_exclusion_reason = None

    return QueryFaceResult(
        faces=faces,
        n_detected=n_detected,
        n_searchable=sum(1 for f in faces if f.vector is not None),
        n_review_only=sum(1 for f in faces if f.vector is None),
        rejected=rejected,
        truncated=truncated,
        cfg=cfg,
    )
