#!/usr/bin/env python3
"""Read-only bounded synthetic-case inventory and restore verifier.

Run inside the release app image with identical /images and /app/data paths.
No inference, model access, face extraction, or database writes are performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request
from qdrant_client import QdrantClient, models


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def inventory(a):
    client = QdrantClient(url=a.qdrant_url, api_key=os.environ.get('SFN_QDRANT_API_KEY'), timeout=30)
    collections = {}
    artifacts = set()
    downloads = {}
    total_points = 0
    image_points = frame_points = video_points = 0
    for name in sorted(c.name for c in client.get_collections().collections):
        rows = []
        offset = None
        while True:
            records, offset = client.scroll(name, limit=100, offset=offset,
                                             with_payload=True, with_vectors=True)
            for r in records:
                total_points += 1
                if total_points > a.max_points:
                    raise RuntimeError('Point bound exceeded; this harness is only for isolated synthetic cases')
                payload = r.payload or {}
                vectors = r.vector or {}
                if not isinstance(vectors, dict):
                    vectors = {'': vectors}
                rows.append({'id': str(r.id), 'payload': payload, 'vectors': vectors})
                if name == a.collection:
                    frame_points += bool(payload.get('is_video_frame'))
                    video_points += bool(payload.get('is_video'))
                    image_points += bool(payload.get('image_path')) and not payload.get('is_video_frame', False)
                for field, route in [('image_path', '/api/hit-image'), ('video_path', '/api/video-download')]:
                    raw = payload.get(field)
                    if raw:
                        path = Path(raw)
                        if not path.is_absolute() or not any(path.is_relative_to(root) for root in a.roots):
                            raise RuntimeError(f'Unexpected artifact path: {raw}')
                        if not path.is_file():
                            raise RuntimeError(f'Missing referenced artifact: {raw}')
                        artifacts.add(path)
                        downloads[str(path)] = route
            if offset is None:
                break
        rows.sort(key=lambda r: r['id'])
        probes = {}
        # Every stored named vector is queried once, using the first point carrying it.
        for row in rows:
            for vector_name, vector in sorted(row['vectors'].items()):
                if vector_name in probes:
                    continue
                response = client.query_points(collection_name=name, query=vector,
                    using=vector_name or None, limit=max(1, len(rows)),
                    search_params=models.SearchParams(exact=True), with_payload=False)
                results = sorted([{'id': str(hit.id), 'score': round(hit.score, 6)}
                                  for hit in response.points], key=lambda hit: hit['id'])
                assert row['id'] in {hit['id'] for hit in results}, 'Query failed to return source point'
                probes[vector_name] = {'source_id': row['id'], 'results': results}
        collections[name] = {'points': rows, 'search_probes': probes}
    assert image_points >= 2, f'Expected at least two synthetic image points, got {image_points}'
    if a.require_video:
        assert frame_points > 0 and video_points > 0, 'Missing indexed video frames or video anchor'
    for root in a.artifact_dirs:
        if root.exists():
            artifacts.update(p for p in root.rglob('*') if p.is_file())
    if len(artifacts) > 2000:
        raise RuntimeError('Artifact bound exceeded')
    checksums = {str(p): digest(p) for p in sorted(artifacts)}
    http_checksums = {}
    if a.http_base:
        for path, route in sorted(downloads.items()):
            url = a.http_base.rstrip('/') + route + '?' + urllib.parse.urlencode({'path': path})
            with urllib.request.urlopen(url, timeout=60) as response:
                assert response.status == 200
                h = hashlib.sha256()
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    h.update(chunk)
            assert h.hexdigest() == checksums[path], f'Original HTTP bytes mismatch: {path}'
            http_checksums[path] = h.hexdigest()
    client.close()
    return {'collections': collections, 'artifacts_sha256': checksums,
            'http_sha256': http_checksums,
            'counts': {'points': total_points, 'images': image_points,
                       'video_frames': frame_points, 'videos': video_points}}


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('mode', choices=['record', 'verify'])
p.add_argument('manifest', type=Path)
p.add_argument('--qdrant-url', default=os.environ.get('SFN_QDRANT_URL', 'http://release-qdrant:6333'))
p.add_argument('--collection', default=os.environ.get('SFN_COLLECTION', 'sfn'))
p.add_argument('--http-base', help='Optional running app URL; use on both record and verify')
p.add_argument('--require-video', action='store_true')
p.add_argument('--max-points', type=int, default=100)
p.add_argument('--roots', type=Path, nargs='+', default=[Path('/images'), Path('/app/data')])
p.add_argument('--artifact-dirs', type=Path, nargs='*',
               default=[Path('/app/data/frames'), Path('/app/data/thumbnails')])
a = p.parse_args()
result = inventory(a)
if a.mode == 'record':
    # Exclusive creation avoids replacing the only baseline during verification.
    with a.manifest.open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2)
        f.write('\n')
else:
    expected = json.loads(a.manifest.read_text())
    if canonical(result) != canonical(expected):
        for key in result:
            if canonical(result[key]) != canonical(expected.get(key)):
                print(f'MISMATCH: {key}')
        raise SystemExit('Restored runtime differs from baseline')
print(json.dumps({'status': 'PASS', 'mode': a.mode, **result['counts'],
                  'artifact_files': len(result['artifacts_sha256']),
                  'http_original_downloads': len(result['http_sha256'])}, sort_keys=True))
