#!/usr/bin/env python3
"""Export tagged reference images to a folder structure for offline review.

Reads all tags from the configured tags collection (default: sfn_tags), resolves
each positive and negative point ID to an image_path (checking the case collection
first, then the reference collection as a fallback), and copies the source images to:

    <output-dir>/<tag-name>/positive/<filename>
    <output-dir>/<tag-name>/negative/<filename>

Filenames are preserved from the source path; collisions are disambiguated with a
short hash suffix so no file is silently dropped.

Usage:
    uv run python scripts/export_reference_collection.py [--url URL] [--collection NAME] …

With no options the script reads SFN_QDRANT_URL, SFN_COLLECTION, SFN_TAGS_COLLECTION
and SFN_REFERENCE_COLLECTION from the environment (or a .env file in the working
directory) and enters an interactive menu.
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: allow running without an editable install
# ---------------------------------------------------------------------------
_src = Path(__file__).parent.parent / "src"
if _src.exists():
    sys.path.insert(0, str(_src))

from qdrant_client import QdrantClient

from scalar_forensic.config import Settings
from scalar_forensic.tags import Tag, TagStore


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def _retrieve_payloads(
    client: QdrantClient,
    case_collection: str,
    reference_collection: str | None,
    point_ids: list[str],
) -> dict[str, dict]:
    """Return {point_id: payload} for each ID, checking case then reference."""
    if not point_ids:
        return {}

    records = list(
        client.retrieve(
            collection_name=case_collection,
            ids=point_ids,
            with_vectors=False,
            with_payload=True,
        )
    )

    found = {str(r.id): r.payload or {} for r in records}

    if reference_collection:
        missing = [i for i in point_ids if i not in found]
        if missing:
            ref_records = client.retrieve(
                collection_name=reference_collection,
                ids=missing,
                with_vectors=False,
                with_payload=True,
            )
            for r in ref_records:
                found[str(r.id)] = r.payload or {}

    return found


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_tag_table(tags: list[Tag]) -> None:
    if not tags:
        print("  (no tags found)")
        return
    print(f"\n  {'Tag name':<40}  {'positive':>8}  {'negative':>8}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}")
    for tag in tags:
        print(
            f"  {tag.name:<40}  {len(tag.positive_ids):>8}  {len(tag.negative_ids):>8}"
        )


# ---------------------------------------------------------------------------
# Export logic
# ---------------------------------------------------------------------------

def _safe_filename(src: Path, dest_dir: Path) -> Path:
    """Return a collision-free destination path under dest_dir."""
    candidate = dest_dir / src.name
    if not candidate.exists():
        return candidate
    short = hashlib.sha256(str(src).encode()).hexdigest()[:8]
    candidate = dest_dir / f"{src.stem}_{short}{src.suffix}"
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        candidate = dest_dir / f"{src.stem}_{short}_{counter}{src.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def do_export(
    client: QdrantClient,
    case_collection: str,
    reference_collection: str | None,
    tags: list[Tag],
    output_dir: Path,
    tag_filter: set[str] | None,
    dry_run: bool,
) -> None:
    """Resolve IDs, preview the plan, and (unless dry_run) copy the files."""
    selected = [t for t in tags if tag_filter is None or t.name in tag_filter]

    if not selected:
        print("  No matching tags.")
        return

    plan: list[tuple[str, str, Path, Path]] = []  # (tag, role, src, dest)
    warnings: list[str] = []

    print(f"\n  Resolving {sum(len(t.positive_ids) + len(t.negative_ids) for t in selected):,}"
          f" point ID(s) across {len(selected)} tag(s) …")

    for tag in selected:
        all_ids = [str(i) for i in tag.positive_ids + tag.negative_ids]
        if not all_ids:
            warnings.append(f"  [warn] tag {tag.name!r} has no point IDs — skipped")
            continue

        payloads = _retrieve_payloads(client, case_collection, reference_collection, all_ids)

        for role, ids in (("positive", tag.positive_ids), ("negative", tag.negative_ids)):
            dest_dir = output_dir / tag.name / role
            for pid in ids:
                payload = payloads.get(str(pid))
                if payload is None:
                    warnings.append(
                        f"  [warn] {tag.name}/{role}: ID {pid} not found in any collection"
                    )
                    continue
                image_path_str = payload.get("image_path", "")
                if not image_path_str:
                    warnings.append(
                        f"  [warn] {tag.name}/{role}: ID {pid} has no image_path"
                    )
                    continue
                src = Path(image_path_str)
                if not src.exists():
                    warnings.append(
                        f"  [warn] {tag.name}/{role}: file not found: {src}"
                    )
                    continue
                dest = _safe_filename(src, dest_dir)
                plan.append((tag.name, role, src, dest))

    # Show plan summary.
    by_tag: dict[str, dict[str, int]] = {}
    for tag_name, role, _src, _dest in plan:
        by_tag.setdefault(tag_name, {"positive": 0, "negative": 0})
        by_tag[tag_name][role] += 1

    print()
    print(f"  Output: {output_dir}\n")
    for tag_name, counts in sorted(by_tag.items()):
        print(
            f"  {tag_name}:  "
            f"{counts['positive']} positive, {counts['negative']} negative"
        )

    if warnings:
        print()
        for w in warnings:
            print(w)

    if not plan:
        print("\n  Nothing to export.")
        return

    print(f"\n  {len(plan)} file(s) will be copied.")

    if dry_run:
        print("\n  [dry-run] No files written.\n")
        return

    answer = input("\n  Proceed? [y/N] ").strip().lower()
    if answer != "y":
        print("  Aborted.")
        return

    copied = 0
    for tag_name, role, src, dest in plan:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"  {tag_name}/{role}/{dest.name}")
        copied += 1

    print(f"\n  Done. {copied} file(s) copied.")


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def run_interactive(
    client: QdrantClient,
    case_collection: str,
    reference_collection: str | None,
    tags_collection: str,
) -> None:
    print("\nLoading tags …")
    store = TagStore(client, tags_collection)
    tags = store.list()
    print(f"  {len(tags)} tag(s) found.")

    while True:
        print("\n" + "─" * 60)
        print("  l  List tags")
        print("  e  Export all tags")
        print("  E  Export all tags (dry-run / preview only)")
        print("  t  Export a specific tag")
        print("  T  Export a specific tag (dry-run / preview only)")
        print("  r  Reload tags from Qdrant")
        print("  q  Quit")
        print("─" * 60)

        choice = input("  > ").strip()

        if choice.lower() == "q":
            break

        elif choice.lower() == "l":
            print_tag_table(tags)

        elif choice.lower() == "r":
            print("Reloading …")
            tags = store.list()
            print(f"  {len(tags)} tag(s) found.")

        elif choice in ("e", "E"):
            dry = choice == "E"
            if not tags:
                print("  No tags available.")
                continue
            out = input("  Output directory: ").strip()
            if not out:
                print("  (empty — cancelled)")
                continue
            try:
                do_export(
                    client=client,
                    case_collection=case_collection,
                    reference_collection=reference_collection,
                    tags=tags,
                    output_dir=Path(out),
                    tag_filter=None,
                    dry_run=dry,
                )
            except Exception as exc:
                print(f"  [error] {exc}")

        elif choice in ("t", "T"):
            dry = choice == "T"
            if not tags:
                print("  No tags available.")
                continue
            print_tag_table(tags)
            names_raw = input("\n  Tag name(s) (comma-separated): ").strip()
            if not names_raw:
                print("  (empty — cancelled)")
                continue
            tag_filter = {n.strip() for n in names_raw.split(",") if n.strip()}
            unknown = tag_filter - {t.name for t in tags}
            if unknown:
                print(f"  Unknown tag(s): {', '.join(sorted(unknown))}")
                if not (tag_filter - unknown):
                    continue
            out = input("  Output directory: ").strip()
            if not out:
                print("  (empty — cancelled)")
                continue
            try:
                do_export(
                    client=client,
                    case_collection=case_collection,
                    reference_collection=reference_collection,
                    tags=tags,
                    output_dir=Path(out),
                    tag_filter=tag_filter,
                    dry_run=dry,
                )
            except Exception as exc:
                print(f"  [error] {exc}")

        else:
            print("  Unknown option.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", help="Qdrant URL (overrides SFN_QDRANT_URL / .env)")
    parser.add_argument("--api-key", help="Qdrant API key (overrides SFN_QDRANT_API_KEY / .env)")
    parser.add_argument(
        "--collection",
        metavar="NAME",
        help="Case collection name (overrides SFN_COLLECTION / .env)",
    )
    parser.add_argument(
        "--tags-collection",
        metavar="NAME",
        help="Tags sidecar collection (overrides SFN_TAGS_COLLECTION / .env)",
    )
    parser.add_argument(
        "--reference-collection",
        metavar="NAME",
        help="Reference collection for fallback ID lookup (overrides SFN_REFERENCE_COLLECTION / .env)",
    )
    args = parser.parse_args()

    settings = Settings()
    url = args.url or settings.qdrant_url
    api_key = args.api_key or settings.qdrant_api_key
    case_collection = args.collection or settings.collection
    tags_collection = args.tags_collection or settings.tags_collection
    reference_collection = args.reference_collection or settings.reference_collection

    print(f"Connecting to Qdrant at {url!r} …")
    client = QdrantClient(url=url, api_key=api_key)

    try:
        existing = {c.name for c in client.get_collections().collections}
    except Exception as exc:
        print(f"ERROR: cannot reach Qdrant: {exc}", file=sys.stderr)
        sys.exit(1)

    if case_collection not in existing:
        print(f"ERROR: case collection {case_collection!r} does not exist.", file=sys.stderr)
        sys.exit(1)

    if tags_collection not in existing:
        print(f"ERROR: tags collection {tags_collection!r} does not exist.", file=sys.stderr)
        sys.exit(1)

    if reference_collection and reference_collection not in existing:
        print(
            f"WARNING: reference collection {reference_collection!r} not found — "
            "IDs will only be looked up in the case collection.",
            file=sys.stderr,
        )
        reference_collection = None

    print(f"  Case collection:      {case_collection!r}")
    print(f"  Tags collection:      {tags_collection!r}")
    print(f"  Reference collection: {reference_collection!r}")

    run_interactive(client, case_collection, reference_collection, tags_collection)


if __name__ == "__main__":
    main()
