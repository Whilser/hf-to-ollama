#!/usr/bin/env python3
import json
import shutil
from pathlib import Path
import sys

import mlx.core as mx

base = Path(sys.argv[1])
source = base / "ollama-weights"
target = base / "normalized-weights"
target.mkdir(exist_ok=True)
index = json.loads((source / "model.safetensors.index.json").read_text())
config = json.loads((source / "config.json").read_text())
overrides = {name for name, value in (config.get("quantization") or config["quantization_config"]).items() if isinstance(value, dict)}
for path in sorted(source.glob("*.safetensors")):
    tensors, metadata = mx.load(str(path), return_metadata=True)
    for name in sorted(overrides):
        weight = name + ".weight"
        if weight not in tensors:
            continue
        scales, biases = name + ".scales", name + ".biases"
        dense = mx.dequantize(tensors[weight], tensors[scales], tensors.pop(biases), group_size=64, bits=8)
        del index["weight_map"][biases]
        if name.endswith(".router.proj"):
            tensors[weight] = dense
            del tensors[scales]
            del index["weight_map"][scales]
        else:
            tensors[weight], tensors[scales] = mx.quantize(dense, group_size=32, bits=4, mode="mxfp4")
    mx.save_safetensors(str(target / path.name), tensors, metadata)
    del tensors
    mx.clear_cache()
    print(path.name, flush=True)
for path in source.iterdir():
    if path.suffix != ".safetensors":
        shutil.copy2(path, target / path.name)
for key in ("quantization", "quantization_config"):
    if key in config:
        config[key] = {name: value for name, value in config[key].items() if name not in overrides}
(target / "config.json").write_text(json.dumps(config))
(target / "model.safetensors.index.json").write_text(json.dumps(index))
(base / "Modelfile.normalized").write_text("FROM ./normalized-weights\n")
