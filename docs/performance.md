# Ingestion Performance

Test run with ~2000 high quality images took about 125,4 seconds using AMD RX 6700 XT (gfx1201) with ROCM, using both embedding models locally.

## Performance metrics

### **Execution Summary**
* **Total Images Processed:** 1,942
* **Non-Image Files Skipped:** 174
* **Total Estimated Time:** ~125.4 seconds (2m 05s)
* **Average Overall Throughput:** ~15.5 images/second
* **Peak Steady-State Throughput:** 42.4 images/second (Batch 36)

### **Performance Metrics by Stage**

| Processing Stage | Est. Total Time | Avg. Time / Batch | Notes |
| :--- | :--- | :--- | :--- |
| **I/O (Read)** | 4.1s | 0.07s | Highly efficient (up to 2.4 GB/s). |
| **Hashing** | 12.8s | 0.21s | MD5/SHA hashing for deduplication. |
| **Preprocessing** | 42.2s | 0.69s | Image resizing and normalization. |
| **SSCD Embedding** | 43.1s | 0.70s | **Primary Bottleneck**; slow in Batch 1 & 61. |
| **DINOv2 Embedding** | 18.9s | 0.31s | Faster inference than SSCD. |
| **Qdrant Upsert** | 3.2s | 0.05s | Database ingestion latency. |
| **Total Pipeline** | **~125.4s** | **2.05s** | Includes all overhead. |

## Runtime complexity

The overall ingestion pipeline has a practical runtime complexity of **$O(N)$**, where $N$ is the total number of files scanned. Because image inputs are capped to a maximum resolution and deep learning models process fixed-size tensors, the processing time per image remains strictly bounded, resulting in linear scaling.

Here is the breakdown of the theoretical complexity by processing stage:

* **File Discovery (Pre-scan):** $O(N)$
  Traversing the directory tree and checking file extensions scales linearly with the total number of files and directories in the target path.
* **I/O & Hashing:** $O(M \cdot S)$
  Reading bytes into memory and computing MD5/SHA-256 hashes scales linearly with the number of valid images ($M$) and their average byte size ($S$).
* **Preprocessing:** $O(M \cdot P)$
  Decoding and resizing images scales linearly with the original pixel count ($P$). However, because the system explicitly caps images to a maximum short-side of 331px, this effectively becomes $O(M)$ constant-time work after the initial downscale.
* **Deduplication Lookups:** $O(M \log V)$
  Querying the Qdrant database to check if hashes or file paths already exist scales logarithmically with the number of previously stored vectors ($V$).
* **Model Inference (SSCD & DINOv2):** $O(M)$
  Neural networks execute a fixed number of operations over fixed-size input tensors (e.g., 288x288 for SSCD, 224x224 for DINOv2). While the underlying mathematical constant is massive (billions of FLOPs) and dictates the primary wall-clock bottleneck, the complexity per image is $O(1)$.
* **Vector Upsert:** $O(M \cdot D)$
  Network transmission and database upserts scale linearly with the embedding dimension ($D$), which is a constant (512 for SSCD, 1024 for DINOv2).

### Summary Formula
The theoretical pipeline complexity is **$O(N + M \cdot (S + P + \log V + D))$**. 

However, because image dimensions ($P$) are capped, embedding sizes ($D$) are constant, and the database lookup time ($\log V$) is negligible compared to inference, the performance scales strictly linearly. For example, if processing 2,000 images takes ~2 minutes, processing 20,000 images of similar composition will predictably take ~20 minutes.

## Real-Life-Dataset

In the below example, we chose to analyze the "unsplash lite" dataset for reference with 25K free-to-use images with above specified hardware. 
The dataset took ~36 minutes.

These are the raw results:

```bash
user01@machine ~/P/ScalarForensic (main)> time uv run sfn /mnt/bulk_storage/sample_images/unsplash_vectors_test_batch/ --dino --sscd
Config: .env
HEIF/HEIC support: disabled (install pillow-heif)
Dedup mode: hash  |  EXIF extraction: True
Loading SSCD model 'models/sscd_disc_mixup.torchscript.pt' on device='auto' ...
  backend=SSCDEmbedder  dim=512  device=cuda  fp16=True  compiled=True
  (first batch will be slow — torch.compile warm-up)
Connecting to Qdrant  collection='sfn-sscd' ...
Computing model hash (may take a moment) ...
  model_hash=9f26bd4c848cc19b...
Loading DINOv2 model 'models/dinov2-large' on device='auto' ...
Loading weights: 100%|███████████| 439/439 [00:00<00:00, 1268.37it/s]
  backend=DINOv2Embedder  dim=1024  device=cuda  fp16=True  compiled=True
  (first batch will be slow — torch.compile warm-up)
Connecting to Qdrant  collection='sfn-dinov2' ...
Computing model hash (may take a moment) ...
  model_hash=9a8b78837fb01923...
Scanning /mnt/bulk_storage/sample_images/unsplash_vectors_test_batch ...
  24,997 files found  (24,997 image, 0 non-image)
  batch 1 [32 imgs  96.6 MB  5.1 img/s total]  read 0.14s (705.7 MB/s)  hash 0.17s  pre 1.90s  |  SSCDEmbedder norm 0.00s  embed 0.54s (59.4 img/s)  +32  |  DINOv2Embedder norm 0.00s  embed 3.48s (9.2 img/s)  +32  |  upsert 0.07s
  batch 2 [32 imgs  76.9 MB  12.5 img/s total]  read 0.12s (663.2 MB/s)  hash 0.13s  pre 1.52s  |  SSCDEmbedder norm 0.02s  embed 0.51s (63.3 img/s)  +32  |  DINOv2Embedder norm 0.00s  embed 0.18s (174.9 img/s)  +32  |  upsert 0.07s
  batch 3 [32 imgs  86.2 MB  12.3 img/s total]  read 0.12s (717.7 MB/s)  hash 0.15s  pre 1.77s  |  SSCDEmbedder norm 0.01s  embed 0.29s (109.3 img/s)  +32  |  DINOv2Embedder norm 0.00s  embed 0.18s (176.7 img/s)  +32  |  upsert 0.05s
  [TRUNCATED]
  batch 779 [32 imgs  99.0 MB  10.8 img/s total]  read 0.13s (739.1 MB/s)  hash 0.17s  pre 1.92s  |  SSCDEmbedder norm 0.00s  embed 0.45s (71.8 img/s)  +32  |  DINOv2Embedder norm 0.00s  embed 0.23s (142.1 img/s)  +32  |  upsert 0.05s
  batch 780 [32 imgs  109.3 MB  12.0 img/s total]  read 0.14s (763.1 MB/s)  hash 0.19s  pre 1.68s  |  SSCDEmbedder norm 0.01s  embed 0.37s (83.3 img/s)  +31  |  DINOv2Embedder norm 0.00s  embed 0.22s (143.6 img/s)  +31  |  upsert 0.05s
  batch 781 [32 imgs  107.5 MB  11.1 img/s total]  read 0.15s (728.1 MB/s)  hash 0.19s  pre 1.89s  |  SSCDEmbedder norm 0.00s  embed 0.36s (88.7 img/s)  +32  |  DINOv2Embedder norm 0.00s  embed 0.21s (151.7 img/s)  +32  |  upsert 0.06s
  batch 782 [5 imgs  50.3 MB  1.7 img/s total]  read 0.06s (865.9 MB/s)  hash 0.09s  pre 2.63s  |  SSCDEmbedder norm 0.01s  embed 0.06s (84.5 img/s)  +5  |  DINOv2Embedder norm 0.00s  embed 0.04s (120.3 img/s)  +5  |  upsert 0.02s

Done.
  [SSCDEmbedder]  indexed=24959  skipped=37  embed_failed=0
  [DINOv2Embedder]  indexed=24959  skipped=37  embed_failed=0

Ingestion summary  (/mnt/bulk_storage/sample_images/unsplash_vectors_test_batch)
────────────────────────────────────────────────────
  Total files found                    24,997
    image files                        24,997
    non-image files (unsupported)           0

  Indexed (new)                        24,959
  Skipped — already indexed                37
  Skipped — duplicate in batch              0
  Failed  — read error                      0
  Failed  — preprocessing                   1
  Failed  — embedding                       0
────────────────────────────────────────────────────
  CSV report → sfn_ingestion_20260412_165320.csv

________________________________________________________
Executed in   36.52 mins    fish           external
```
