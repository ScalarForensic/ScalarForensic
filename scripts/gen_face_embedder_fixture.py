"""One-shot generator for the tiny ONNX embedder fixture (run once, commit output).

Run with the export-only deps supplied ad hoc (they are NOT project deps):
    uv run --with onnx --with onnxscript python scripts/gen_face_embedder_fixture.py
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUT = Path("tests/fixtures/faces")


class _TinyEmbedder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 512, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).mean(dim=(2, 3))  # global average pool -> (N, 512)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tiny_face.onnx"
    model = _TinyEmbedder().eval()
    # external_data=False keeps weights inside the .onnx: a sidecar .data file
    # would sit outside hash_file(model_path)'s provenance coverage.
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, 112, 112),),
        str(path),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        external_data=False,
    )
    manifest = {
        "input_name": "input",
        "layout": "NCHW",
        "channel_order": "RGB",
        "dtype": "float32",
        "input_size": 112,
        "mean": 127.5,
        "scale": 128.0,
        "output_name": "embedding",
        "embedding_dim": 512,
    }
    Path(str(path) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"fixture written to {path}")

    # Fixed-batch-1 variant: same net, no dynamic axes — mirrors real detector
    # exports like SFace whose input is declared (1, 3, 112, 112).  Used to test
    # that OnnxFaceEmbedder chunks multi-face batches instead of crashing.
    path_b1 = OUT / "tiny_face_batch1.onnx"
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, 112, 112),),
        str(path_b1),
        input_names=["input"],
        output_names=["embedding"],
        external_data=False,
    )
    Path(str(path_b1) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"fixture written to {path_b1}")


if __name__ == "__main__":
    main()
