"""Face indexing orchestration (spec §5).

Pure orchestration: decode -> detect -> pre-gate -> align -> post-gate ->
embed (one batch per image) -> chips -> points.  Qdrant I/O stays at the
edge: process_image() returns points and the caller upserts them, which
keeps the core testable without a server.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from scalar_forensic.faces.align import ALIGNMENT_VERSION, align_face
from scalar_forensic.faces.audit import AuditLog
from scalar_forensic.faces.chips import write_chips
from scalar_forensic.faces.decode import load_for_detection
from scalar_forensic.faces.detect import YuNetDetector
from scalar_forensic.faces.embed import OnnxFaceEmbedder
from scalar_forensic.faces.provenance import PipelineConfig
from scalar_forensic.faces.quality import post_align_gate, pre_align_gate
from scalar_forensic.faces.store import FACE_VECTOR_NAME, FaceStore


@dataclass
class FaceIndexResult:
    n_detected: int = 0
    n_kept: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    points: list[PointStruct] = field(default_factory=list)


@dataclass
class FacePipeline:
    detector: object
    embedder: object
    store: FaceStore
    audit: AuditLog
    cfg: PipelineConfig
    min_conf: float
    min_size: int
    min_sharpness: float
    max_clipped: float
    max_pose: float
    crop_dilation: float
    store_dir: Path | None
    thumb_size: int = 256

    @classmethod
    def from_settings(cls, settings) -> FacePipeline:
        err = settings.face_startup_error()
        if err:
            raise ValueError(err)
        detector = YuNetDetector(settings.face_detector_model, settings.face_detect_max_size)
        embedder = OnnxFaceEmbedder(settings.face_embedder_model)
        cfg = PipelineConfig(
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
            sfn_version=_sfn_version(),
            cv2_version=cv2.__version__,
            ort_version=ort.__version__,
        )
        store = FaceStore(
            QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
            settings.face_collection,
            settings.collection,
            embedder.manifest.embedding_dim,
        )
        store_dir = Path(settings.face_store_dir) if settings.face_store_dir else None
        audit_dir = store_dir.parent if store_dir else Path("data")
        return cls(
            detector=detector,
            embedder=embedder,
            store=store,
            audit=AuditLog(audit_dir / "face_audit.log"),
            cfg=cfg,
            min_conf=settings.face_min_conf,
            min_size=settings.face_min_size,
            min_sharpness=settings.face_min_sharpness,
            max_clipped=settings.face_max_clipped,
            max_pose=settings.face_max_pose,
            crop_dilation=settings.face_crop_dilation,
            store_dir=store_dir,
            thumb_size=settings.face_thumb_size,
        )

    def _composite_quality(self, pre_subs: dict, post_subs: dict) -> float:
        """Bootstrap composite in [0, 1]; 0 means "at the gate boundary".

        Each subscore is normalised against its own gate threshold so the
        number is comparable across differently-configured runs.  This is a
        browse/ranking aid, not evidence — the raw subscores are stored
        alongside it and the Phase 1b calibration record supersedes the
        formula.
        """
        parts = [
            pre_subs["confidence"],
            1.0 - pre_subs["pose"] / self.max_pose if self.max_pose else 1.0,
            min(1.0, post_subs["sharpness"] / (2 * self.min_sharpness))
            if self.min_sharpness
            else 1.0,
            1.0 - post_subs["exposure"] / self.max_clipped if self.max_clipped else 1.0,
        ]
        return float(np.clip(min(parts), 0.0, 1.0))

    def process_image(
        self,
        data: bytes,
        image_hash: str,
        image_path: str,
        video_hash: str | None = None,
        video_path: str | None = None,
        frame_timecode_ms: int | None = None,
    ) -> FaceIndexResult:
        img = load_for_detection(data)
        detections = self.detector.detect(img)
        result = FaceIndexResult(n_detected=len(detections))
        if not detections:
            return result

        kept: list[tuple] = []  # (det, aligned, pre_subs, post_subs)
        for det in detections:
            pre = pre_align_gate(
                det, min_conf=self.min_conf, min_size=self.min_size, max_pose=self.max_pose
            )
            if not pre.passed:
                result.rejected[pre.reason] = result.rejected.get(pre.reason, 0) + 1
                continue
            aligned = align_face(img, det.landmarks)
            # Sharpness/exposure are measured on the native-resolution source
            # crop, not the 112x112 resample (spec §6.2).
            x, y, w, h = (int(v) for v in det.bbox)
            crop = img[max(0, y) : y + h, max(0, x) : x + w]
            if crop.size == 0:
                result.rejected["size"] = result.rejected.get("size", 0) + 1
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            post = post_align_gate(
                gray, min_sharpness=self.min_sharpness, max_clipped_frac=self.max_clipped
            )
            if not post.passed:
                result.rejected[post.reason] = result.rejected.get(post.reason, 0) + 1
                continue
            kept.append((det, aligned, pre.subscores, post.subscores))

        if not kept:
            return result

        embeddings = self.embedder.embed([k[1] for k in kept])
        norms = np.asarray(self.embedder.embedding_norms)
        provenance = self.cfg.to_payload()
        indexed_at = datetime.now(UTC).isoformat()

        for i, (det, aligned, pre_subs, post_subs) in enumerate(kept):
            chip_hash = None
            if self.store_dir is not None:
                chip_hash = write_chips(
                    self.store_dir,
                    aligned,
                    img,
                    bbox=det.bbox,
                    dilation=self.crop_dilation,
                    thumb_size=self.thumb_size,
                )
            payload = {
                "is_face": True,
                "image_hash": image_hash,
                "image_path": image_path,
                "video_hash": video_hash,
                "video_path": video_path,
                "frame_timecode_ms": frame_timecode_ms,
                "observation_key": self.store.observation_key(
                    image_hash, frame_timecode_ms, det.bbox
                ),
                "bbox": [int(round(v)) for v in det.bbox],
                "landmarks": det.landmarks.tolist(),
                "det_conf": det.confidence,
                "detect_scale": det.detect_scale,
                "quality_confidence": pre_subs["confidence"],
                "quality_size": pre_subs["size"],
                "quality_pose": pre_subs["pose"],
                "quality_sharpness": post_subs["sharpness"],
                "quality_exposure": post_subs["exposure"],
                "quality": self._composite_quality(pre_subs, post_subs),
                "embedding_norm": float(norms[i]) if i < len(norms) else 0.0,
                "chip_hash": chip_hash,
                "indexed_at": indexed_at,
                **provenance,
            }
            result.points.append(
                PointStruct(
                    id=self.store.face_point_id(image_hash, frame_timecode_ms, det.bbox),
                    vector={FACE_VECTOR_NAME: [float(v) for v in embeddings[i]]},
                    payload=payload,
                )
            )
        result.n_kept = len(result.points)
        return result


def _sfn_version() -> str:
    try:
        return importlib.metadata.version("scalar-forensic")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - dev checkout
        return "unknown"
