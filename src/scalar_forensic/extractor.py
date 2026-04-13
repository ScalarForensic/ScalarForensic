"""Container file extraction — produces ExtractedImage records for ingestion."""

from __future__ import annotations

import hashlib
import os
import os
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image as PilImage

# File extensions recognised as container formats (hold images inside).
CONTAINER_EXTENSIONS: frozenset[str] = frozenset({".zip", ".docx", ".odt", ".odp", ".ods", ".pdf"})

# Image formats that may appear inside containers.
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".gif",
        ".jp2",
        ".psd",
        ".ico",
        ".heif",
        ".heic",
    }
)


@dataclass
class ExtractedImage:
    """One image extracted from a container file."""

    data: bytes
    """Raw image bytes."""

    item_name: str
    """Full internal path from the root container, using ``::`` as the level separator.

    Examples: ``"photos/shot.jpg"``, ``"inner.zip::photos/shot.jpg"``,
    ``"page_1"``, ``"page_1::embed_0"``.
    """

    parent_hash: str
    """SHA-256 of the *immediate* parent.

    For top-level children this equals the SHA-256 of the root container file.
    For embedded PDF images this equals the SHA-256 of the rendered page PNG.
    """

    parent_type: str
    """Type of the immediate parent: ``"zip"``, ``"docx"``, ``"odf"``,
    ``"pdf"``, or ``"pdf_page"``."""

    root_container_path: Path
    """Absolute filesystem path to the root (outermost) container file."""

    root_container_type: str
    """Type of the root container: ``"zip"``, ``"docx"``, ``"odf"``, or ``"pdf"``."""

    extraction_kind: str
    """How the image was obtained: ``"rendered"`` (PDF page rasterised) or
    ``"embedded"`` (image blob extracted directly)."""


def extract_container(
    path: Path,
    *,
    max_depth: int = 5,
    pdf_render_dpi: int = 150,
    allowed_root: Path,
) -> list[ExtractedImage]:
    """Extract all images from a container file (ZIP / DOCX / ODF / PDF).

    Returns a flat list of :class:`ExtractedImage` objects.  Nested containers
    (e.g. a ZIP inside a ZIP) are processed recursively up to *max_depth* levels.

    :param path: Filesystem path to the container file.
    :param max_depth: Maximum nesting depth.  Depth 1 means only the root
        container is processed; depth 2 allows one level of nesting, etc.
    :param pdf_render_dpi: Resolution used when rasterising PDF pages.
    :param allowed_root: Trusted root directory.  *path* must resolve inside
        this directory; an error is raised if it does not.
    resolved_root = allowed_root.resolve(strict=True)
    path = path.resolve()
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"Container path is not a file: {path}") from exc

    if os.path.commonpath((str(resolved_root), str(resolved_path))) != str(resolved_root):
        raise ValueError(f"Container path is outside allowed root: {resolved_path}")

    path = resolved_path
    if not path.is_file():
        raise ValueError(f"Container path is outside allowed root: {path}") from exc
    # Reconstruct from the trusted root so subsequent ops use a path whose
    # provenance is anchored to the validated allowed_root.
    path = resolved_root / _rel
    if not path.exists() or not path.is_file():
        raise ValueError(f"Container path is not a file: {path}")
    ext = path.suffix.lower()
    if ext not in CONTAINER_EXTENSIONS:
        raise ValueError(f"Unsupported container extension: {ext}")
    data = path.read_bytes()
    container_hash = _sha256(data)
    root_type = _ext_to_type(ext)

    return _dispatch(
        data=data,
        ext=ext,
        item_prefix="",
        parent_hash=container_hash,
        root_path=path,
        root_type=root_type,
        depth=1,
        max_depth=max_depth,
        pdf_render_dpi=pdf_render_dpi,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ext_to_type(ext: str) -> str:
    mapping = {
        ".zip": "zip",
        ".pdf": "pdf",
        ".docx": "docx",
        ".odt": "odf",
        ".odp": "odf",
        ".ods": "odf",
    }
    return mapping.get(ext, "unknown")


def _is_valid_image(data: bytes) -> bool:
    """Return True if *data* can be opened as a PIL image."""
    try:
        with PilImage.open(BytesIO(data)) as img:
            img.verify()
        return True
    except Exception:
        return False


def _dispatch(
    *,
    data: bytes,
    ext: str,
    item_prefix: str,
    parent_hash: str,
    root_path: Path,
    root_type: str,
    depth: int,
    max_depth: int,
    pdf_render_dpi: int,
) -> list[ExtractedImage]:
    if ext == ".zip":
        return _extract_zip(
            data, item_prefix, parent_hash, root_path, root_type, depth, max_depth, pdf_render_dpi
        )
    if ext == ".pdf":
        return _extract_pdf(
            data, item_prefix, parent_hash, root_path, root_type, depth, max_depth, pdf_render_dpi
        )
    if ext == ".docx":
        return _extract_docx(
            data, item_prefix, parent_hash, root_path, root_type, depth, max_depth, pdf_render_dpi
        )
    if ext in {".odt", ".odp", ".ods"}:
        return _extract_odf(
            data, item_prefix, parent_hash, root_path, root_type, depth, max_depth, pdf_render_dpi
        )
    return []


def _extract_zip(
    zip_bytes: bytes,
    item_prefix: str,
    parent_hash: str,
    root_path: Path,
    root_type: str,
    depth: int,
    max_depth: int,
    pdf_render_dpi: int,
) -> list[ExtractedImage]:
    results: list[ExtractedImage] = []
    try:
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue  # directory entry
                suffix = Path(name).suffix.lower()
                item_name = f"{item_prefix}{name}" if item_prefix else name
                try:
                    member_bytes = zf.read(name)
                except Exception:
                    continue

                if suffix in _IMAGE_EXTENSIONS and _is_valid_image(member_bytes):
                    results.append(
                        ExtractedImage(
                            data=member_bytes,
                            item_name=item_name,
                            parent_hash=parent_hash,
                            parent_type="zip",
                            root_container_path=root_path,
                            root_container_type=root_type,
                            extraction_kind="embedded",
                        )
                    )
                elif suffix in CONTAINER_EXTENSIONS and depth < max_depth:
                    inner_hash = _sha256(member_bytes)
                    nested = _dispatch(
                        data=member_bytes,
                        ext=suffix,
                        item_prefix=f"{item_name}::",
                        parent_hash=inner_hash,
                        root_path=root_path,
                        root_type=root_type,
                        depth=depth + 1,
                        max_depth=max_depth,
                        pdf_render_dpi=pdf_render_dpi,
                    )
                    results.extend(nested)
    except zipfile.BadZipFile:
        # Invalid or corrupt ZIP payload — treat as non-extractable container.
        return results
    return results


def _extract_docx(
    docx_bytes: bytes,
    item_prefix: str,
    parent_hash: str,
    root_path: Path,
    root_type: str,
    depth: int,  # noqa: ARG001
    max_depth: int,  # noqa: ARG001
    pdf_render_dpi: int,  # noqa: ARG001
) -> list[ExtractedImage]:
    """Extract embedded images from a DOCX file via python-docx."""
    results: list[ExtractedImage] = []
    try:
        try:
            from docx import Document  # noqa: PLC0415
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "python-docx is required for DOCX extraction: "
                "install it with 'uv sync --group containers'"
            ) from None

        doc = Document(BytesIO(docx_bytes))
        seen_parts: set[int] = set()
        for rel in doc.part.rels.values():
            if "image" not in rel.reltype:
                continue
            try:
                part = rel.target_part
            except Exception:
                continue
            part_id = id(part)
            if part_id in seen_parts:
                continue
            seen_parts.add(part_id)
            img_bytes = part.blob
            # partname is like "/word/media/image1.png"
            part_name = part.partname.lstrip("/")
            item_name = f"{item_prefix}{part_name}" if item_prefix else part_name
            if _is_valid_image(img_bytes):
                results.append(
                    ExtractedImage(
                        data=img_bytes,
                        item_name=item_name,
                        parent_hash=parent_hash,
                        parent_type="docx",
                        root_container_path=root_path,
                        root_container_type=root_type,
                        extraction_kind="embedded",
                    )
                )
    except Exception:
        # Best-effort extraction: unreadable or malformed DOCX must not abort
        # container processing; return whatever images were extracted so far.
        return results
    return results


def _extract_odf(
    odf_bytes: bytes,
    item_prefix: str,
    parent_hash: str,
    root_path: Path,
    root_type: str,
    depth: int,  # noqa: ARG001
    max_depth: int,  # noqa: ARG001
    pdf_render_dpi: int,  # noqa: ARG001
) -> list[ExtractedImage]:
    """Extract images from an ODF container (ODT / ODS / ODP).

    ODF files are ZIP archives; images are stored under ``Pictures/``.
    """
    results: list[ExtractedImage] = []
    try:
        with zipfile.ZipFile(BytesIO(odf_bytes)) as zf:
            for name in zf.namelist():
                if not name.startswith("Pictures/") or name.endswith("/"):
                    continue
                suffix = Path(name).suffix.lower()
                if suffix not in _IMAGE_EXTENSIONS:
                    continue
                try:
                    img_bytes = zf.read(name)
                except Exception:
                    continue
                item_name = f"{item_prefix}{name}" if item_prefix else name
                if _is_valid_image(img_bytes):
                    results.append(
                        ExtractedImage(
                            data=img_bytes,
                            item_name=item_name,
                            parent_hash=parent_hash,
                            parent_type="odf",
                            root_container_path=root_path,
                            root_container_type=root_type,
                            extraction_kind="embedded",
                        )
                    )
    except zipfile.BadZipFile:
        # Invalid or corrupt ODF archive — treat as non-extractable container.
        return results
    return results


def _extract_pdf(
    pdf_bytes: bytes,
    item_prefix: str,
    parent_hash: str,
    root_path: Path,
    root_type: str,
    depth: int,  # noqa: ARG001
    max_depth: int,  # noqa: ARG001
    pdf_render_dpi: int,
) -> list[ExtractedImage]:
    """Extract images from a PDF.

    For each page two sets of images are produced:

    1. A **rendered** PNG raster of the full page (child of the PDF).
    2. **Embedded** image blobs found on that page (children of the rendered page).
    """
    try:
        import fitz  # noqa: PLC0415  (pymupdf)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PDF extraction requires the optional PyMuPDF dependency. "
            "Install it with 'uv sync --group containers'."
        ) from exc

    results: list[ExtractedImage] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return results

    zoom = pdf_render_dpi / 72.0  # pymupdf default resolution is 72 DPI
    mat = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_label = f"page_{page_num + 1}"
        page_item_name = f"{item_prefix}{page_label}" if item_prefix else page_label

        # 1. Render page to PNG.
        try:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            page_png = pix.tobytes("png")
        except Exception:
            continue

        results.append(
            ExtractedImage(
                data=page_png,
                item_name=page_item_name,
                parent_hash=parent_hash,
                parent_type="pdf",
                root_container_path=root_path,
                root_container_type=root_type,
                extraction_kind="rendered",
            )
        )

        # 2. Extract embedded image blobs — children of the rendered page.
        rendered_page_hash = _sha256(page_png)
        try:
            image_list = page.get_images(full=True)
        except Exception:
            image_list = []

        seen_xrefs: set[int] = set()
        for embed_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
            except Exception:
                continue
            if not img_bytes or not _is_valid_image(img_bytes):
                continue
            img_ext = base_image.get("ext") or "png"
            embed_item_name = f"{page_item_name}::embed_{embed_idx}.{img_ext}"
            results.append(
                ExtractedImage(
                    data=img_bytes,
                    item_name=embed_item_name,
                    parent_hash=rendered_page_hash,
                    parent_type="pdf_page",
                    root_container_path=root_path,
                    root_container_type=root_type,
                    extraction_kind="embedded",
                )
            )

    doc.close()
    return results
