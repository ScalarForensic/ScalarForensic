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
