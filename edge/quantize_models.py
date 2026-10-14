"""Make smaller MLX copies of the LlamaPIE models, so the live pipeline fits in 16 GB.

    generator  (8B): 8-bit -> 6-bit, output layer kept at 8-bit   (8.0 GB -> 6.2 GB)
    classifier (1B): fp16  -> 8-bit, 2-class head kept at fp16    (2.3 GB -> 1.2 GB)

On the sample dialogues these give the same whispers as the originals; a 4-bit
generator (--generator-bits 4, 4.5 GB) is faster but loses some memory details.

    python quantize_models.py          # writes edge/models/
"""
import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_lm.utils import get_model_path, load, quantize_model, save_config, save_weights

from proactive_agent import BIG_MODEL, SMALL_MODEL

MODELS_DIR = Path(__file__).parent / "models"
SMALL_8BIT = MODELS_DIR / "classifier-1b-8bit"


def generator_path(bits):
    return MODELS_DIR / f"generator-8b-{bits}bit"


def dequantize(model):
    """Turn quantized linear/embedding layers back into float16 ones (lazily)."""
    layers = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            w = mx.dequantize(m.weight, m.scales, m.biases, m.group_size, m.bits).astype(mx.float16)
            if isinstance(m, nn.QuantizedEmbedding):
                new = nn.Embedding(*w.shape)
            else:
                new = nn.Linear(w.shape[1], w.shape[0], bias="bias" in m)
                if "bias" in m:
                    new.bias = m.bias
            new.weight = w
            layers.append((name, new))
    model.update_modules(tree_unflatten(layers))
    return model


def convert(repo, out, bits, keep):
    """Re-quantize `repo` to `bits`; layers in `keep` get their own setting (False = float)."""
    if out.exists():
        print(f"{out} already exists, skipping")
        return
    src = get_model_path(repo)
    model, _ = load(repo, lazy=True)
    config = json.loads((src / "config.json").read_text())
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    model = dequantize(model)

    def predicate(path, module, _config):
        if not hasattr(module, "to_quantized"):
            return False
        return keep.get(path, True)

    weights, config = quantize_model(model, config, 64, bits, quant_predicate=predicate)
    out.mkdir(parents=True)
    for f in src.iterdir():
        if f.suffix != ".safetensors" and f.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(f, out / f.name)
    save_weights(out, weights)
    save_config(config, out / "config.json")
    size = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 2**30
    print(f"Wrote {out} ({size:.1f} GB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator-bits", type=int, choices=[4, 5, 6], default=6)
    args = parser.parse_args()
    convert(SMALL_MODEL, SMALL_8BIT, bits=8, keep={"classifier": False})
    convert(BIG_MODEL, generator_path(args.generator_bits), bits=args.generator_bits,
            keep={"lm_head": {"group_size": 64, "bits": 8}})


if __name__ == "__main__":
    main()
