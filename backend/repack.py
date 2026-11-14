#!/usr/bin/env python3
import json
import mmap
import shutil
import struct
from pathlib import Path
import sys

base = Path(sys.argv[1])
source = base / "weights"
target = base / "ollama-weights"
target.mkdir(exist_ok=True)
index_path = source / "model.safetensors.index.json"
index = json.loads(index_path.read_text())
new_map = {}
stem = ".experts.switch_glu."
for path in sorted(source.glob("*.safetensors")):
    print(path.name, flush=True)
    with path.open("rb") as infile, mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as data:
        header_length = struct.unpack_from("<Q", data)[0]
        start = 8 + header_length
        tensors = json.loads(data[8:start])
        metadata = tensors.pop("__metadata__", None)
        new_header = {}
        chunks = []
        offset = 0
        for name, tensor in tensors.items():
            names = [name]
            shape = tensor["shape"].copy()
            output_name = name
            if stem + "up_proj." in name:
                continue
            if stem + "gate_proj." in name:
                peer = name.replace(stem + "gate_proj.", stem + "up_proj.")
                assert tensors[peer]["shape"] == shape
                assert tensors[peer]["dtype"] == tensor["dtype"]
                output_name = name.replace(stem + "gate_proj.", ".experts.gate_up_proj.")
                names.append(peer)
                shape[1] *= 2
            elif stem in name:
                assert stem + "down_proj." in name
                output_name = name.replace(stem, ".experts.")
            spans = [tensors[key]["data_offsets"] for key in names]
            size = sum(end - begin for begin, end in spans)
            assert output_name not in new_map
            new_map[output_name] = path.name
            new_header[output_name] = dict(dtype=tensor["dtype"], shape=shape, data_offsets=[offset, offset + size])
            chunks.append((spans, tensor["shape"][0]))
            offset += size
        if metadata is not None:
            new_header["__metadata__"] = metadata
        header = json.dumps(new_header, separators=(",", ":")).encode()
        header += b" " * (-len(header) % 8)
        with (target / path.name).open("wb") as out:
            out.write(struct.pack("<Q", len(header)))
            out.write(header)
            for spans, experts in chunks:
                if len(spans) == 1:
                    begin, end = spans[0]
                    for pos in range(begin, end, 8 * 1024 * 1024):
                        out.write(data[start + pos:start + min(end, pos + 8 * 1024 * 1024)])
                else:
                    for i in range(experts):
                        for begin, end in spans:
                            width, remainder = divmod(end - begin, experts)
                            assert remainder == 0
                            pos = start + begin + i * width
                            out.write(data[pos:pos + width])
for path in source.iterdir():
    if path.is_file() and path.suffix != ".safetensors" and path != index_path:
        shutil.copy2(path, target / path.name)
index["weight_map"] = new_map
(target / index_path.name).write_text(json.dumps(index))
(base / "Modelfile.repacked").write_text("FROM ./ollama-weights\n")
