# Deployment topologies

ScalarForensic is designed so that its four roles can run on one machine or be
distributed across several machines on an isolated local network, reached over
their local IPs:

| Role | Process | Listens on | Key settings |
|---|---|---|---|
| Vector database | Qdrant | `6333` | — |
| Embedding compute (GPU) | any OpenAI-compatible embeddings server | your choice | — |
| Application host | `sfn` (CLI indexing) / `sfn-web` (web UI) | `SFN_WEB_PORT` (default `8080`) | everything below |
| Clients | browser | — | just the app host URL |

All configuration is environment variables (or `.env` next to the process CWD);
precedence is process env → `.env` → defaults. See `.env.example` for the full
annotated list.

## Topology 1 — single machine (default)

Everything on one workstation. The defaults work as-is: Qdrant on
`http://localhost:6333`, models loaded locally (GPU/CPU per `SFN_DEVICE`),
web UI on `http://localhost:8080`.

## Topology 2 — separate Qdrant host

Move storage/database load to another machine. On the application host:

```bash
SFN_QDRANT_URL=http://192.168.10.20:6333
SFN_QDRANT_API_KEY=...        # if the Qdrant instance requires it
```

The application host still computes embeddings locally, so it needs the model
files and (ideally) a GPU.

## Topology 3 — full split (Qdrant + GPU embedding server + app host)

Embedding compute moves to a dedicated GPU machine exposing an
OpenAI-compatible embeddings API. On the application host:

```bash
SFN_QDRANT_URL=http://192.168.10.20:6333

SFN_EMBEDDING_ENDPOINT=http://192.168.10.30:9000/v1/embeddings
SFN_EMBEDDING_API_KEY=...     # if the endpoint requires it
SFN_EMBEDDING_MODEL=<model name the endpoint expects>
SFN_EMBEDDING_DIM=<embedding dimension the endpoint returns>
```

Note the integrity trade-off: a remote endpoint cannot produce a weights hash,
only a config hash derived from endpoint + model name + dimension. These are
stored and displayed with a `config:` prefix and a "cfg-only" chip in the UI so
they are never mistaken for weights hashes. For court-facing work, prefer
locally verified models (Topology 1/2) or pin the remote server's model
artefact out-of-band.

## Reachability matrix

- Clients → application host: `SFN_WEB_PORT` (default `8080`).
- Application host → Qdrant: `SFN_QDRANT_URL`.
- Application host → embedding server: `SFN_EMBEDDING_ENDPOINT` (Topology 3 only).
- Nothing needs to reach the clients; Qdrant and the embedding server never
  talk to each other.
- No internet access is required or expected at runtime: the network policy is
  offline-by-default (`SFN_ALLOW_ONLINE=false`). Set it to `true` (or pass
  `--allow-online`) only for first-time model downloads, then revert.

`sfn-web` binds `SFN_WEB_HOST=0.0.0.0` by default so LAN clients can reach it.
On a machine that is not inside the isolated network, narrow it to
`SFN_WEB_HOST=127.0.0.1`.

## State that lives on the application host

These paths are read/written by both `sfn` (indexing) and `sfn-web` (serving),
so **indexing and the web UI must run on the same machine — or share these
directories** (e.g. one mount) if you split them:

| Setting | Default | Used for |
|---|---|---|
| `SFN_INPUT_DIR` | unset | evidence images root; also the allowed root for file-serving endpoints |
| `SFN_THUMBNAIL_DIR` | `data/thumbnails` | thumbnails written during indexing, served at `/api/thumbnail/…` |
| `SFN_FRAME_STORE_DIR` | `data/frames` | extracted video frames, source for thumbnail regeneration |
| `SFN_HASH_CACHE_PATH` | `data/hash_cache.db` | SHA-256 cache keyed by (path, mtime, size) — path-specific, do not share between machines |
| ingestion reports | `data/reports/` | per-run CSV reports from `sfn index` |

Everything else (collections, tags, reference material) lives in Qdrant and
moves with it.

## Multi-user limits (web UI)

The web UI supports modest concurrent use out of the box:

- `SFN_MAX_ACTIVE_SESSIONS` (default 32) — further sessions get HTTP 503.
- `SFN_SESSION_TTL_SECONDS` (default 3600) — idle sessions are reaped every 60 s.
- `SFN_MAX_UPLOAD_BYTES` (default 2 GiB) — per-request upload cap, HTTP 413 above.

## Consistency constraints across machines

Embeddings are only comparable when produced with identical parameters. The
startup/ingest safeguards will fail fast on mismatches, but when distributing,
keep these identical everywhere embeddings are produced:

- model artefacts (`SFN_MODEL_DINO`, `SFN_MODEL_SSCD`) — verify with the stored
  model hash (`scripts/download_models.py --hash <stored_hash>`)
- `SFN_NORMALIZE_SIZE`
- `SFN_SSCD_N_CROPS`

Collection names (`SFN_COLLECTION`, `SFN_TAGS_COLLECTION`,
`SFN_REFERENCE_COLLECTION`) must match on every machine that talks to the same
Qdrant instance.

## Legal assumptions for the optional face modality

This section states **assumptions, not legal conclusions.**

The face modality is a post-hoc biometric search aid operated by the deploying organisation
(the deployer role). Depending on jurisdiction and use, EU AI Act obligations for post-remote
biometric identification, GDPR/LED duties (including a DPIA and Art. 9/10 special-category
processing), and national law — including authorisation regimes for law-enforcement use — may
apply, with obligations phasing in on the Act's timeline.

**The technical controls in ScalarForensic support such compliance; they do not establish it.**
Jurisdiction-specific legal review before enablement is an operator duty, as is the assessment
of your own role under the applicable framework.

The controls the legal story rests on:

- **Append-only audit log** (`data/face_audit.log`) recording enablement, index runs and
  purges — what the system did and how it was used, separately from artefact provenance.
- **Enablement record**: first activation writes examiner id, timestamp and a free-text
  authorization reference into the collection metadata and the audit log. Enabling the feature
  is a recorded act, not config drift.
- **Provenance on every artefact**: models, hashes, alignment version, and every gate threshold
  in force (see `docs/face-matching-math.md`).
- **Purge path** for face observations and their chip files.
- **Disabled by default**, with no bundled recognition weights.

Machine output is an investigative lead. Identification is a human conclusion by a named
examiner, and the tool is built to keep that line visible.
