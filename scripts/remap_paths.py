#!/usr/bin/env python3
"""Interactively remap file paths stored in the Qdrant collections.

When evidence drives are moved or mounted at different paths, the stored
``image_path`` and ``video_path`` payload fields go stale.  This tool
enumerates every distinct path in the collection(s), shows you how many
points each one affects, and lets you replace prefixes in bulk.

Usage:
    uv run python scripts/remap_paths.py [--url URL] [--collection NAME]

With no arguments it reads SFN_QDRANT_URL / SFN_COLLECTION_* from the
environment (or a .env file in the working directory) and operates on
both configured collections.

Point updates are performed via ``set_payload`` filtered by the exact
old path value — payload indexes on ``image_path`` and ``video_path``
make this efficient even for large collections.
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: allow running without an editable install
# ---------------------------------------------------------------------------
_src = Path(__file__).parent.parent / "src"
if _src.exists():
    sys.path.insert(0, str(_src))

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from scalar_forensic.config import Settings

# Separator used in virtual video-frame paths.
_FRAME_SEP = "::"


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def _scroll_all(client: QdrantClient, collection: str) -> list[dict]:
    """Return the payload of every point in *collection* (no vectors)."""
    points: list[dict] = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            limit=500,
            offset=offset,
            with_payload=["image_path", "video_path", "is_video_frame"],
            with_vectors=False,
        )
        for p in batch:
            if p.payload:
                points.append({"id": p.id, **p.payload})
        if next_offset is None:
            break
        offset = next_offset
    return points


def enumerate_paths(
    client: QdrantClient, collections: list[str]
) -> dict[str, dict[str, dict]]:
    """Return path statistics grouped by field name and collection.

    Returns::

        {
          "image_path": {
              "<collection>": {"<path>": <count>, ...},
              ...
          },
          "video_path": {
              "<collection>": {"<path>": <count>, ...},
          },
        }
    """
    result: dict[str, dict[str, dict]] = {
        "image_path": {},
        "video_path": {},
    }

    existing = {c.name for c in client.get_collections().collections}

    for coll in collections:
        if coll not in existing:
            print(f"  [skip] collection {coll!r} does not exist")
            continue

        print(f"  Scanning {coll!r} ...", end=" ", flush=True)
        points = _scroll_all(client, coll)
        print(f"{len(points):,} points")

        img_counts: dict[str, int] = {}
        vid_counts: dict[str, int] = {}

        for p in points:
            ip = p.get("image_path")
            vp = p.get("video_path")
            if ip:
                img_counts[ip] = img_counts.get(ip, 0) + 1
            if vp:
                vid_counts[vp] = vid_counts.get(vp, 0) + 1

        result["image_path"][coll] = img_counts
        result["video_path"][coll] = vid_counts

    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _merge_counts(by_collection: dict[str, dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for counts in by_collection.values():
        for path, n in counts.items():
            merged[path] = merged.get(path, 0) + n
    return merged


def print_path_table(stats: dict[str, dict[str, dict]], field: str, label: str) -> None:
    by_coll = stats.get(field, {})
    merged = _merge_counts(by_coll)
    if not merged:
        print(f"  (no {label} paths found)")
        return
    print(f"\n  {label} paths ({len(merged):,} unique):")
    for path in sorted(merged):
        total = merged[path]
        per_coll = ", ".join(
            f"{c}: {by_coll[c][path]}" for c in sorted(by_coll) if path in by_coll[c]
        )
        print(f"    [{total:>6}]  {path}  ({per_coll})")


def print_path_table_by_dir(stats: dict[str, dict[str, dict]], field: str, label: str) -> None:
    """Aggregate point counts by parent directory instead of individual file."""
    by_coll = stats.get(field, {})
    merged = _merge_counts(by_coll)
    if not merged:
        print(f"  (no {label} paths found)")
        return

    dir_counts: dict[str, int] = {}
    for path, n in merged.items():
        # Virtual video-frame paths (image_path field) embed the video path
        # before "::"; use the directory of the video file, not the suffix.
        real_path = path.split(_FRAME_SEP)[0] if _FRAME_SEP in path else path
        parent = str(Path(real_path).parent)
        dir_counts[parent] = dir_counts.get(parent, 0) + n

    print(f"\n  {label} paths by directory ({len(dir_counts):,} unique dirs, {sum(dir_counts.values()):,} points):")
    for d in sorted(dir_counts):
        print(f"    [{dir_counts[d]:>6}]  {d}/")


# ---------------------------------------------------------------------------
# Remap
# ---------------------------------------------------------------------------

def _paths_matching_prefix(
    by_coll: dict[str, dict[str, int]], prefix: str
) -> dict[str, list[str]]:
    """Return {collection: [matching_paths]} for paths starting with *prefix*."""
    result: dict[str, list[str]] = {}
    for coll, counts in by_coll.items():
        matches = [p for p in counts if p.startswith(prefix)]
        if matches:
            result[coll] = matches
    return result


def _update_image_paths(
    client: QdrantClient,
    collection: str,
    old_paths: list[str],
    old_prefix: str,
    new_prefix: str,
) -> int:
    """Replace ``old_prefix`` with ``new_prefix`` in ``image_path`` for each old path."""
    updated = 0
    for old in old_paths:
        new = new_prefix + old[len(old_prefix):]
        client.set_payload(
            collection_name=collection,
            payload={"image_path": new},
            points=Filter(
                must=[FieldCondition(key="image_path", match=MatchValue(value=old))]
            ),
        )
        updated += 1
    return updated


def _update_video_paths(
    client: QdrantClient,
    collection: str,
    old_paths: list[str],
    old_prefix: str,
    new_prefix: str,
) -> int:
    """Replace ``old_prefix`` with ``new_prefix`` in ``video_path`` *and* ``image_path``.

    For video frames, ``image_path`` is a virtual path that embeds the video
    path as its prefix before the ``::`` separator.  Both fields must stay in sync.
    """
    updated = 0
    for old_vid in old_paths:
        new_vid = new_prefix + old_vid[len(old_prefix):]
        # Update video_path and — for frames that share this video — image_path too.
        # We filter by video_path so we catch all frames at once.
        client.set_payload(
            collection_name=collection,
            payload={"video_path": new_vid},
            points=Filter(
                must=[FieldCondition(key="video_path", match=MatchValue(value=old_vid))]
            ),
        )
        # image_path for video frames looks like:  /old/video.mp4::frame_...
        # We replace only the part before ::.
        old_img_prefix = old_vid + _FRAME_SEP
        new_img_prefix = new_vid + _FRAME_SEP
        # Scroll all frames for this video to patch image_path individually.
        frame_points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="video_path", match=MatchValue(value=new_vid))]
            ),
            limit=10_000,
            with_payload=["image_path"],
            with_vectors=False,
        )
        for fp in frame_points:
            old_ip = (fp.payload or {}).get("image_path", "")
            if old_ip.startswith(old_img_prefix):
                new_ip = new_img_prefix + old_ip[len(old_img_prefix):]
                client.set_payload(
                    collection_name=collection,
                    payload={"image_path": new_ip},
                    points=Filter(
                        must=[FieldCondition(key="image_path", match=MatchValue(value=old_ip))]
                    ),
                )
        updated += 1
    return updated


def _snapshot_collections(client: QdrantClient, collections: list[str]) -> None:
    """Create a Qdrant snapshot for each collection before mutating data."""
    print()
    for coll in collections:
        print(f"  Snapshotting {coll!r} ...", end=" ", flush=True)
        try:
            info = client.create_snapshot(collection_name=coll, wait=True)
            print(f"ok  →  {info.name}")
        except Exception as exc:
            print(f"FAILED ({exc})")
            raise SystemExit(
                "Snapshot failed — aborting to protect data. "
                "Check Qdrant storage permissions or disk space."
            ) from exc


def do_remap(
    client: QdrantClient,
    stats: dict[str, dict[str, dict]],
    old_prefix: str,
    new_prefix: str,
    dry_run: bool,
) -> None:
    """Execute (or preview) a prefix remap across all tracked fields."""

    img_matches = _paths_matching_prefix(stats["image_path"], old_prefix)
    vid_matches = _paths_matching_prefix(stats["video_path"], old_prefix)

    # For video frames the image_path starts with the video path, so image_path
    # matches are a superset.  Separate "pure" image paths (no video_path match)
    # from virtual frame paths so we display them cleanly.
    all_old_img: dict[str, list[str]] = {}  # coll -> [img paths excluding virtual frames]
    for coll, paths in img_matches.items():
        pure = [p for p in paths if _FRAME_SEP not in p]
        if pure:
            all_old_img[coll] = pure

    total_img_points = sum(
        stats["image_path"][c].get(p, 0)
        for c, paths in all_old_img.items()
        for p in paths
    )
    total_vid_files = sum(len(v) for v in vid_matches.values())
    total_vid_frames = sum(
        stats["video_path"][c].get(p, 0)
        for c, paths in vid_matches.items()
        for p in paths
    )

    if not all_old_img and not vid_matches:
        print("  No paths match that prefix.")
        return

    print(f"\n  Prefix:  {old_prefix!r}")
    print(f"  Replace: {new_prefix!r}\n")

    if all_old_img:
        print(f"  Standalone image paths affected: {total_img_points:,} points")
        for coll, paths in sorted(all_old_img.items()):
            for p in sorted(paths)[:10]:
                new_p = new_prefix + p[len(old_prefix):]
                print(f"    {p}")
                print(f"      → {new_p}")
            if len(paths) > 10:
                print(f"    … and {len(paths) - 10} more")

    if vid_matches:
        print(f"\n  Video files affected: {total_vid_files:,} ({total_vid_frames:,} frames)")
        for coll, paths in sorted(vid_matches.items()):
            for p in sorted(paths)[:10]:
                new_p = new_prefix + p[len(old_prefix):]
                print(f"    {p}")
                print(f"      → {new_p}")
            if len(paths) > 10:
                print(f"    … and {len(paths) - 10} more")

    if dry_run:
        print("\n  [dry-run] No changes written.")
        return

    print()
    answer = input("  Apply these changes? [y/N] ").strip().lower()
    if answer != "y":
        print("  Aborted.")
        return

    affected_collections = sorted(
        {c for c in list(all_old_img) + list(vid_matches)}
    )
    _snapshot_collections(client, affected_collections)

    total_updated = 0

    for coll, paths in all_old_img.items():
        n = _update_image_paths(client, coll, paths, old_prefix, new_prefix)
        total_updated += n
        print(f"  Updated {n} image_path value(s) in {coll!r}")

    for coll, paths in vid_matches.items():
        n = _update_video_paths(client, coll, paths, old_prefix, new_prefix)
        total_updated += n
        print(f"  Updated {n} video_path value(s) (+ frame image_paths) in {coll!r}")

    print(f"\n  Done. {total_updated} path value(s) remapped.")


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def run_interactive(client: QdrantClient, collections: list[str]) -> None:
    print("\nLoading path index …")
    stats = enumerate_paths(client, collections)

    while True:
        print("\n" + "─" * 60)
        print("  l  List all paths (verbose)")
        print("  d  List paths aggregated by directory")
        print("  r  Remap a prefix")
        print("  R  Remap a prefix (dry-run / preview only)")
        print("  q  Quit")
        print("─" * 60)

        choice = input("  > ").strip().lower()

        if choice == "q":
            break

        elif choice == "l":
            print_path_table(stats, "image_path", "image")
            print_path_table(stats, "video_path", "video")

        elif choice == "d":
            print_path_table_by_dir(stats, "image_path", "image")
            print_path_table_by_dir(stats, "video_path", "video")

        elif choice in ("r", "R"):
            dry = choice == "R"
            old = input("  Old prefix: ").strip()
            if not old:
                print("  (empty — cancelled)")
                continue
            new = input("  New prefix: ").strip()
            if not new:
                print("  (empty — cancelled)")
                continue
            do_remap(client, stats, old, new, dry_run=dry)
            if not dry:
                # Refresh stats after a live update.
                print("\nRefreshing path index …")
                stats = enumerate_paths(client, collections)

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
    parser.add_argument(
        "--url",
        help="Qdrant URL (overrides SFN_QDRANT_URL / .env)",
    )
    parser.add_argument(
        "--collection",
        metavar="NAME",
        action="append",
        help=(
            "Collection to operate on (repeat for multiple). "
            "Defaults to both SFN_COLLECTION_SSCD and SFN_COLLECTION_DINO."
        ),
    )
    parser.add_argument(
        "--api-key",
        help="Qdrant API key (overrides SFN_QDRANT_API_KEY / .env)",
    )
    args = parser.parse_args()

    settings = Settings()
    url = args.url or settings.qdrant_url
    api_key = args.api_key or settings.qdrant_api_key
    collections = args.collection or [settings.collection_sscd, settings.collection_dino]

    print(f"Connecting to Qdrant at {url!r} …")
    client = QdrantClient(url=url, api_key=api_key)

    # Quick connectivity check.
    try:
        client.get_collections()
    except Exception as exc:
        print(f"ERROR: cannot reach Qdrant: {exc}", file=sys.stderr)
        sys.exit(1)

    run_interactive(client, collections)


if __name__ == "__main__":
    main()
