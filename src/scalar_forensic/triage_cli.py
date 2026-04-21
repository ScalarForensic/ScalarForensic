"""Command-line driver for Concept-Triage (Qdrant Discovery) queries.

Exposed as the ``sfn-triage`` console script.  Covers the operator
workflow that does not need the web UI: create / list / inspect
concepts, mark references, and run triage runs against the case
collection, optionally writing a JSONL report compatible with the
existing ``sfn --report`` format.

Concepts are persisted in the sidecar Qdrant collection configured by
``SFN_CONCEPTS_COLLECTION`` (default ``sfn_concepts``) — see
:mod:`scalar_forensic.concepts`.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from qdrant_client import QdrantClient

from scalar_forensic.concepts import ConceptStore
from scalar_forensic.config import Settings
from scalar_forensic.discovery import run_triage

app = typer.Typer(
    add_completion=False,
    help="Concept-Triage: investigator-in-the-loop Discovery queries.",
    no_args_is_help=True,
)


def _open_store(settings: Settings) -> tuple[QdrantClient, ConceptStore]:
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return client, ConceptStore(client, settings.concepts_collection)


def _parse_id_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


@app.command("create")
def create_cmd(
    name: str = typer.Argument(..., help="Human-readable concept name (stable ID key)."),
    positive: str = typer.Option(
        "", "--positive", "-p", help="Comma-separated Qdrant point IDs marked positive."
    ),
    negative: str = typer.Option(
        "", "--negative", "-n", help="Comma-separated Qdrant point IDs marked negative."
    ),
    target: str = typer.Option(
        "", "--target", "-t", help="Optional single target anchor point ID."
    ),
    polarity: str = typer.Option(
        "incriminating",
        "--polarity",
        help="'incriminating' (default) or 'exculpatory' — metadata only.",
    ),
    notes: str = typer.Option("", "--notes", help="Free-form description."),
) -> None:
    """Create or replace a concept by name (name-derived UUID is stable)."""
    if polarity not in ("incriminating", "exculpatory"):
        raise typer.BadParameter("polarity must be 'incriminating' or 'exculpatory'")
    settings = Settings()
    _client, store = _open_store(settings)
    concept = store.create(
        name,
        positive_ids=_parse_id_list(positive),
        negative_ids=_parse_id_list(negative),
        target_id=target.strip() or None,
        polarity=polarity,  # type: ignore[arg-type]
        notes=notes,
    )
    typer.echo(json.dumps(
        {
            "concept_id": concept.concept_id,
            "name": concept.name,
            "positives": len(concept.positive_ids),
            "negatives": len(concept.negative_ids),
            "target": concept.target_id,
            "polarity": concept.polarity,
        },
        indent=2,
    ))


@app.command("list")
def list_cmd() -> None:
    """List every concept in the sidecar collection, newest first."""
    settings = Settings()
    _client, store = _open_store(settings)
    for c in store.list():
        typer.echo(
            f"{c.concept_id}  {c.name!r:30}  "
            f"pos={len(c.positive_ids):>3}  neg={len(c.negative_ids):>3}  "
            f"polarity={c.polarity}  updated={c.updated_at}"
        )


@app.command("show")
def show_cmd(concept_id: str = typer.Argument(...)) -> None:
    settings = Settings()
    _client, store = _open_store(settings)
    concept = store.get(concept_id)
    if concept is None:
        raise typer.Exit(code=1)
    typer.echo(json.dumps(concept.to_payload(), indent=2, default=str))


@app.command("mark")
def mark_cmd(
    concept_id: str = typer.Argument(...),
    point_id: str = typer.Argument(..., help="Qdrant point ID to label."),
    role: str = typer.Option(
        "positive", "--role", "-r", help="'positive' or 'negative'."
    ),
) -> None:
    if role not in ("positive", "negative"):
        raise typer.BadParameter("role must be 'positive' or 'negative'")
    settings = Settings()
    _client, store = _open_store(settings)
    concept = store.mark(concept_id, point_id, role)  # type: ignore[arg-type]
    typer.echo(
        f"Concept {concept.concept_id}: "
        f"pos={len(concept.positive_ids)} neg={len(concept.negative_ids)}"
    )


@app.command("delete")
def delete_cmd(concept_id: str = typer.Argument(...)) -> None:
    settings = Settings()
    _client, store = _open_store(settings)
    existed = store.delete(concept_id)
    raise typer.Exit(code=0 if existed else 1)


@app.command("run")
def run_cmd(
    concept_id: str = typer.Argument(..., help="Concept ID returned by 'create'."),
    mode: str = typer.Option(
        "dual", "--mode", help="'dual' (default), 'dino', or 'sscd'."
    ),
    limit: int = typer.Option(50, "--limit", "-l", min=1, max=1000),
    reverse: bool = typer.Option(
        False,
        "--reverse",
        help=(
            "Swap positive/negative roles at query time.  Use to surface "
            "material the concept places on the BENIGN side of the boundary "
            "(exculpatory triage), e.g. to auto-hide from review."
        ),
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        help="Write one JSON object per hit to this file (JSONL format).",
    ),
) -> None:
    """Run a Concept-Triage query and print (or write) ranked hits."""
    if mode not in ("dual", "dino", "sscd"):
        raise typer.BadParameter("mode must be 'dual', 'dino', or 'sscd'")
    settings = Settings()
    client, store = _open_store(settings)
    concept = store.get(concept_id)
    if concept is None:
        typer.echo(f"Concept not found: {concept_id}", err=True)
        raise typer.Exit(code=1)

    hits = run_triage(
        client,
        settings.collection,
        concept,
        mode=mode,  # type: ignore[arg-type]
        limit=limit,
        reverse=reverse,
        reference_collection=settings.reference_collection,
    )

    if report is not None:
        with report.open("w", encoding="utf-8") as fh:
            for h in hits:
                fh.write(
                    json.dumps(
                        {
                            "concept_id": concept.concept_id,
                            "point_id": h.point_id,
                            "matched_modes": h.matched_modes,
                            "triplet_score_dino": h.triplet_score_dino,
                            "triplet_score_sscd": h.triplet_score_sscd,
                            "cosine_margin_dino": h.cosine_margin_dino,
                            "cosine_margin_sscd": h.cosine_margin_sscd,
                            "fused_triplet_score": h.fused_triplet_score,
                            "fused_cosine_margin": h.fused_cosine_margin,
                            "path": (h.payload or {}).get("image_path"),
                            "image_hash": (h.payload or {}).get("image_hash"),
                            "is_video_frame": bool(
                                (h.payload or {}).get("is_video_frame")
                            ),
                            "video_path": (h.payload or {}).get("video_path"),
                            "frame_timecode_ms": (h.payload or {}).get(
                                "frame_timecode_ms"
                            ),
                        },
                        default=str,
                    )
                    + "\n"
                )
        typer.echo(f"Wrote {len(hits)} hits to {report}")
        return

    for h in hits:
        modes = ",".join(h.matched_modes) or "-"
        typer.echo(
            f"{h.fused_triplet_score:>3}  [{modes:>9}]  "
            f"margin={h.fused_cosine_margin:.3f}  "
            f"{(h.payload or {}).get('image_path') or h.point_id}"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
