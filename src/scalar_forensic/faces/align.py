"""Umeyama 5-point similarity alignment to the ArcFace 112x112 template.

Implementation reference: Umeyama (1991), "Least-squares estimation of
transformation parameters between two point patterns" — NOT the research
corpus (its equations were stripped in export; spec §6.4).  Reference
points verified against insightface.utils.face_align.arcface_dst.
warpAffine parameters are pinned: bilinear, BORDER_CONSTANT black.
"""

from __future__ import annotations

import cv2
import numpy as np

ALIGNMENT_VERSION = "arcface-112-v1"
ALIGNED_SIZE = 112

ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation+scale+translation), as 2x3."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = dst_c.T @ src_c / n
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[1, 1] = -1  # reflection correction — a face alignment must never mirror
    rot = u @ s @ vt
    var_src = (src_c**2).sum() / n
    scale = float(np.trace(np.diag(d) @ s) / var_src)
    m = np.zeros((2, 3))
    m[:, :2] = scale * rot
    m[:, 2] = mu_dst - scale * rot @ mu_src
    return m


def align_face(img_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Warp *img_rgb* so *landmarks* (canonical order) land on ARCFACE_DST."""
    m = umeyama(landmarks, ARCFACE_DST)
    return cv2.warpAffine(
        img_rgb,
        m,
        (ALIGNED_SIZE, ALIGNED_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
