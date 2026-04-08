# ScalarForensic

Large-scale image vector indexer for forensic use. Embeds images into a Qdrant vector database for similarity search and deduplication.

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)
- Qdrant: `docker run -p 6333:6333 qdrant/qdrant`

## Setup

```bash
git clone https://github.com/ScalarForensic/ScalarForensic
cd ScalarForensic
uv sync
```

## Usage

```bash
uv run sfn index <image-dir> --sscd
uv run sfn index <image-dir> --dino
```

### Backends

| Flag | Model | Dim | Best for |
|------|-------|-----|----------|
| `--sscd` | SSCD ResNet-50 (Meta) | 512 | Copy detection, near-duplicates |
| `--dino` | DINOv2 ViT-L (Meta) | 1024 | Semantic similarity |

### SSCD setup (one-time)

SSCD is not on HuggingFace — download the checkpoint manually:

```bash
wget https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt
```

Place it in the project directory or pass `--model /path/to/sscd_disc_mixup.torchscript.pt`.  
DINOv2 downloads automatically on first run.

### Options

```
--model        Override model path / HuggingFace identifier
--collection   Qdrant collection name (default: sfn-sscd or sfn-dinov2)
--batch-size   Images per batch (default: 32)
--device       auto | cuda | cpu | mps (default: auto)
--qdrant-url   Qdrant server URL (default: http://localhost:6333)
```
