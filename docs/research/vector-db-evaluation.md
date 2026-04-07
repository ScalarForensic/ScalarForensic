# Vector Database Evaluation

## Evaluation Criteria
1. Memory footprint with quantization at 500M vector scale (highest priority)
2. HNSW index recall accuracy after quantization
3. Single-node operational simplicity
4. Python client maturity
5. Active maintenance

---

## Milvus

- **Memory footprint at scale:** For 1M 128-dim float32 vectors, HNSW alone uses ~640 MB (128 MB graph + 512 MB vectors). With PQ at 8 subquantizers (8 bytes/vector), the combined HNSW_PQ index drops to ~136 MB per million vectors (128 MB graph + 8 MB compressed vectors), representing a 64x compression over raw vectors. Scaled to 500M vectors at 512 dimensions, raw storage would be ~1 TB; HNSW_PQ compresses the vector portion drastically, but the HNSW graph structure itself still scales proportionally with vector count and remains fully in-memory. Milvus 2.6 added native INT8 vector support (HNSW family) for ingesting quantized embeddings directly. A documented test with 500M 512-dim vectors achieved ~5 QPS at 1700ms average latency with 10 concurrent users — suggesting significant memory pressure on a single node at this scale. **Note on benchmark asymmetry:** This 500M-vector stress test is a known published data point for Milvus; no equivalent publicly documented stress test at this exact scale exists for Qdrant or Weaviate. Comparisons at 500M vectors should therefore be treated with caution — Milvus's performance at this scale is directly observed, while Qdrant and Weaviate figures are extrapolated from smaller-scale benchmarks and vendor documentation.
- **HNSW + PQ recall:** HNSW on SIFT dataset achieves: 85.5% (ef=16), 93.9% (ef=32), 98.1% (ef=64), 99.5% (ef=128), 99.9% (ef=256+). HNSW_PQ trades some recall for compression; HNSW_PRQ (Product Residual Quantizer) achieves higher recall than HNSW_PQ at equivalent compression. SQ8 scalar quantization reduces memory 75% with minor recall loss and is generally preferred over PQ for recall preservation. Milvus has the highest throughput at recall below 0.95; gap narrows above 0.95.
- **Single-node ops:** Milvus Standalone requires Docker Compose with three co-located containers: milvus-standalone, milvus-etcd, and milvus-minio. The milvus.yaml configuration file has 500+ parameters. etcd requires NVMe SSDs with >500 IOPS and <10ms p99 fsync latency. This is significantly more operational overhead than Qdrant or Weaviate for single-node use. Milvus Lite (pip install, embedded) exists for dev/testing but is not suitable for 500M-vector production workloads. Known pain points include etcd stability under heavy load, MinIO configuration for production security (default keys must be changed), and complexity when tuning query/index node separation.
- **Python client:** Official `pymilvus` library with full async support. Well-documented on milvus.io with extensive tutorials. Supports ORM-style and MilvusClient simplified API. Mature and widely used by enterprise teams (NVIDIA, Salesforce, eBay, Airbnb, DoorDash). Active on PyPI.
- **Maintenance:** 40,000+ GitHub stars (surpassed milestone in late 2025, the fastest growth spurt in project history). Active releases: Milvus 2.5 (hybrid search), 2.6 (INT8 native support, cost/performance improvements). Security release 2.5.27 issued for CVE-2026-26190 (critical authentication bypass, CVSS 9.8; NVD: https://nvd.nist.gov/vuln/detail/CVE-2026-26190). 10,000+ enterprise teams in production. 24/7 global support, Milvus Ambassadors program, rewrote documentation. Backed by Zilliz with managed cloud (Zilliz Cloud).
- **Sources:**
  - https://milvus.io/docs/benchmark.md
  - https://github.com/milvus-io/milvus/discussions/19189
  - https://milvus.io/ai-quick-reference/how-much-memory-overhead-is-typically-introduced-by-indexes-like-hnsw-or-ivf-for-a-given-number-of-vectors-and-how-can-this-overhead-be-managed-or-configured
  - https://milvus.io/docs/index.md
  - https://milvus.io/docs/prerequisite-docker.md
  - https://milvus.io/docs/install_standalone-docker-compose.md
  - https://milvus.io/blog/milvus-exceeds-40k-github-stars.md
  - https://medium.com/@tspann/how-good-is-quantization-in-milvus-6d224b5160b0 (community observation, not independently verified)
  - https://medium.com/@Nexumo_/10-reproducible-benchmarks-for-milvus-qdrant-weaviate-02723160b89d (community observation, not independently verified)

---

## Qdrant

- **Memory footprint at scale (note: scalar quantization vs PQ):** Qdrant's primary production quantization is scalar quantization (float32 → int8), achieving 4x memory compression. The formula for raw HNSW memory is: `1.5 × N × dim × 4 bytes`. For 500M vectors at 512 dim, raw = ~1.5 TB; with scalar quantization (4x compression), the quantized vector portion drops to ~375 GB. Critically, Qdrant supports a hybrid mmap strategy: quantized vectors and HNSW index in RAM, original vectors on disk (on_disk=True, always_ram=True for quantized). This allows Qdrant to index and serve 400M vectors on a single 64 GB RAM machine, with original vectors accessed from disk only for rescoring top candidates. At 500M vectors, a machine with 80–96 GB RAM is plausible with this strategy. PQ is also available but noted as slower than scalar quantization (PQ distance calculations are not SIMD-friendly) and with higher accuracy penalties; it is recommended only when memory is the absolute top priority and search speed is secondary.
  - **Scalar vs PQ distinction:** Scalar quantization maintains 99%+ recall accuracy and is SIMD-optimized, delivering up to 2x speed improvement. PQ provides higher compression (potentially >4x) but compounds approximation errors across sub-vectors, leading to more significant accuracy penalties and slower queries. For this use case (high recall required), scalar quantization is the superior choice in Qdrant.
- **Quantization approach and recall:** Scalar quantization achieves 99%+ recall in production across diverse embedding models. Binary quantization (1-bit) achieves 32x compression with 4x speed gains on models specifically trained for it (e.g., OpenAI Ada-002, Cohere multilingual), but requires rescoring to maintain high recall. Qdrant 1.15 introduced 1.5-bit, 2-bit, and asymmetric quantization options. GPU-accelerated HNSW indexing was added in 2025 for faster ingestion. HNSW graph compression (in-memory footprint reduction for the graph itself) was also introduced in 2025.
- **Single-node ops:** Qdrant is a single binary written in Rust, deployable as one Docker container with no external dependencies. It runs on a $20/mo VPS handling millions of vectors. No etcd, no MinIO, no message queues. Configuration is straightforward via YAML or REST API. Consistently rated as the simplest vector database to self-host among the three candidates. On-disk storage for original vectors with mmap reduces hardware requirements dramatically. Qdrant also supports embedded mode for development.
- **Python client:** Official `qdrant-client` on PyPI, well-maintained. Supports both sync and async from version 1.6.1. Type definitions for all Qdrant API methods. Active release cadence on GitHub (qdrant/qdrant-client). Extensive documentation at qdrant.tech. Integrations with LangChain, LlamaIndex, and 35 additional integrations added in 2025.
- **Maintenance:** 27,000+ GitHub stars as of 2025. Active Rust-based development with frequent releases. Discord community at 8,000+ members. 35 new integrations in 2025. Qdrant 2025 recap highlighted GPU indexing, inline storage, HNSW graph compression, and smarter quantization. TurboQuant quantization (ICLR 2026) in consideration. Cloud offering at cloud.qdrant.io.
- **Known limitations:**
  - **Single-node scalability ceiling:** While Qdrant's mmap strategy extends single-node viability significantly, at 500M+ vectors with high RPS requirements, single-node deployments can hit hardware ceilings. Vertical scaling has natural limits, and single-node mode offers no redundancy. Horizontal scaling via sharding is supported but has a constraint: one shard cannot be split across nodes, meaning shard count must be planned upfront — adding nodes to an under-sharded cluster cannot fully utilize the new capacity without resharding. Resharding (changing shard count in-place) is available but adds operational complexity.
  - **Production bug history at scale:** Released versions have carried notable production bugs including an integer overflow in query batching at high limits, a corrupted ID tracker when disk fills, cancellation failures under heavy search load, and OOM-inducing parallel segment loading on large clusters. These were addressed in patch releases (tracked on GitHub releases), but indicate that large-scale production use requires careful version management and monitoring. Teams operating near memory limits should pin to well-tested patch releases rather than immediately adopting new minor versions.
  - **No built-in visualization or admin UI:** Qdrant ships without a native dashboard or visualization layer. Operational observability relies on external tooling (Prometheus metrics, Grafana dashboards, or third-party tools). This adds setup overhead for teams accustomed to databases with built-in management consoles.
  - **Multi-tenancy fragility at extreme tenant counts:** While payload-based filtering supports multi-tenant workloads, using the same point ID across multiple shard keys is unsupported and can cause data inconsistency. At very high tenant counts, multi-tenancy patterns can become fragile without careful schema design.
- **Sources:**
  - https://qdrant.tech/articles/memory-consumption/
  - https://qdrant.tech/articles/scalar-quantization/
  - https://qdrant.tech/articles/product-quantization/
  - https://qdrant.tech/course/essentials/day-4/large-scale-ingestion/
  - https://qdrant.tech/benchmarks/
  - https://qdrant.tech/benchmarks/single-node-speed-benchmark/
  - https://qdrant.tech/articles/vector-search-resource-optimization/
  - https://qdrant.tech/blog/qdrant-1.15.x/
  - https://qdrant.tech/blog/2025-recap/
  - https://github.com/orgs/qdrant/discussions/2607
  - https://github.com/qdrant/qdrant-client

---

## Weaviate

- **Memory footprint at scale:** Memory rule of thumb: `2 × (memory footprint of all vectors)`. For 1M 384-dim float32 vectors, ~3 GB total. For 500M 512-dim vectors, uncompressed = 500M × 512 × 4 bytes = ~1 TB raw vectors; with the 2x overhead formula, ~2 TB is needed without compression. With PQ compression (up to 90% reduction), vector memory can drop significantly — e.g., 768-dim vectors compressed from 3072 bytes to 128 bytes (24x). However, the HNSW graph structure is not compressed by PQ and must remain fully in RAM; graph size is approximately N × maxConnections × 10 bytes. For 500M vectors at maxConnections=64, the graph alone is ~320 GB. Weaviate's 8-bit Rotational Quantization (RQ, added in v1.32) maintains 98–99% recall with 4x compression. Teams report resource pressure above 100M vectors; at 500M vectors on a single node, Weaviate requires very high-spec hardware. Weaviate is written in Go (not Java), which is more memory-efficient than Java but still heavier than Rust (Qdrant) at equivalent loads.
- **HNSW + PQ recall:** PQ can reduce memory by up to 90% but makes tradeoffs against recall and performance. Weaviate mitigates recall loss via rescoring: it overfetches candidates from the compressed index and rescores against uncompressed vectors. RQ (v1.32) achieves 98–99% recall in internal testing with 4x compression and minimal configuration. PQ recall depends heavily on segment and centroid configuration and requires training data (10,000+ vectors). BlockMax WAND (v1.34) improves BM25 hybrid search latency by 94% but does not directly affect vector recall.
- **Single-node ops:** Weaviate runs as a single Docker container and is generally easier to deploy than Milvus Standalone. However, it requires more memory and compute than Qdrant at large scale. Weaviate's module system (vectorizer modules, generative modules) adds configuration complexity if used; for pure vector storage/search (bring-your-own embeddings), it is manageable. Resource usage becomes a real concern above 100M vectors. Multiple shards are recommended even on a single node for import performance. The v1.34 release adds 30+ new observability metrics for Prometheus/Grafana, indicating maturing operational tooling. Overall: simpler than Milvus, more complex than Qdrant, with higher resource requirements than both at scale.
- **Python client:** Official `weaviate-client` (v4) on PyPI for Python 3.9+. Well-documented. Active releases (weaviate/weaviate-python-client). A separate `weaviate-agents-python-client` exists for agent workflows. v4 client introduced synchronous and async support. Weaviate also has new C# and Java clients as of v1.34. GraphQL interface is available alongside the Python client but adds complexity for pure vector search use cases.
- **Maintenance:** GitHub: weaviate/weaviate is active with regular releases. v1.34 released in 2025 with preview features (flat index RQ, server-side batch imports), ACORN filter strategy as default, and expanded observability. Weaviate Cloud Services available. Community forum active. ~12,000+ GitHub stars (lower than both Milvus and Qdrant as of 2025 comparisons, though exact current count not confirmed in these results). Backed by Weaviate B.V. with enterprise support.
- **Sources:**
  - https://weaviate.io/developers/weaviate/concepts/resources
  - https://weaviate.io/blog/ann-algorithms-hnsw-pq
  - https://docs.weaviate.io/weaviate/concepts/vector-quantization
  - https://weaviate.io/blog/pq-rescoring
  - https://weaviate.io/blog/binary-quantization
  - https://weaviate.io/blog/weaviate-1-34-release
  - https://github.com/weaviate/weaviate
  - https://github.com/weaviate/weaviate-python-client
  - https://medium.com/@fendylike/top-5-open-source-vector-search-engines-a-comprehensive-comparison-guide-for-2025-e10110b47aa3 (community observation, not independently verified)

---

## Recommendation

**Winner:** Qdrant

**Reasoning:** Qdrant is the clear choice for single-node deployment at 500M-vector scale with quantization. Its mmap-based hybrid strategy (quantized vectors in RAM, original vectors on disk) is the only architecture among the three that makes 400–500M vectors viable on a single node with 64–96 GB RAM — a documented and supported production pattern. Scalar quantization delivers 4x compression with 99%+ recall accuracy and SIMD-optimized speed, which is superior to PQ in terms of recall preservation for workloads where false negatives are costly. Single-binary Rust deployment requires no external dependencies (no etcd, no MinIO), making it the lowest operational overhead of the three.

**Runner-up:** Milvus — viable if the deployment environment already has Kubernetes or Docker Swarm infrastructure and the team can absorb etcd/MinIO operational overhead; Milvus offers superior indexing throughput and richer index options (HNSW_PRQ) if hardware resources are unconstrained.

**Rejected:** Weaviate — because its HNSW graph must remain fully in RAM and the graph alone for 500M vectors at typical maxConnections is estimated at 300+ GB, making single-node deployment at this scale extremely hardware-intensive; teams consistently report resource pressure above 100M vectors, and it delivers 4x lower RPS than Qdrant at equivalent recall thresholds in benchmarks (this figure originates from a Qdrant-published benchmark at qdrant.tech/benchmarks and should be treated as a vendor claim pending independent corroboration).

**Operational requirements note:** Access control, audit logging, and backup/restore capabilities were not evaluated in this comparison and must be assessed separately before deployment in a production evidence context. All three candidates offer some form of these capabilities: Qdrant provides role-based access control and payload-level filtering with API key authentication (see https://qdrant.tech/documentation/guides/security/); Milvus provides RBAC, TLS, and audit logging via the Milvus security framework (see https://milvus.io/docs/authenticate.md and https://milvus.io/docs/rbac.md); Weaviate provides API key and OIDC authentication, backup/restore via its backup module, and structured access control (see https://weaviate.io/developers/weaviate/configuration/authentication and https://weaviate.io/developers/weaviate/configuration/backups). These capabilities must be validated against the specific access control policies, chain-of-custody requirements, and audit standards applicable to the evidence management environment before any production deployment decision is finalized.
