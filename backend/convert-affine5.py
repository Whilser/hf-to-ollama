import json,shutil,sys
from pathlib import Path
import mlx.core as mx
source=Path(sys.argv[1])
target=Path(sys.argv[2])
target.mkdir(exist_ok=True)
for path in source.glob('*.safetensors'):
 tensors,metadata=mx.load(str(path),return_metadata=True)
 for key in list(tensors):
  if not key.endswith('.scales'): continue
  base=key[:-7]
  weight=base+'.weight'
  bias=base+'.biases'
  dense=mx.dequantize(tensors[weight],tensors[key],tensors.get(bias),group_size=32,bits=5,mode='affine')
  tensors[weight],tensors[key],tensors[bias]=mx.quantize(dense,group_size=32,bits=8,mode='affine')
 mx.save_safetensors(str(target/path.name),tensors,metadata)
 del tensors
 mx.clear_cache()
 print(path.name,flush=True)
for path in source.iterdir():
 if path.is_file() and path.suffix!='.safetensors': shutil.copy2(path,target/path.name)
config=json.loads((target/'config.json').read_text())
for key in ('quantization','quantization_config'): config[key]={'group_size':32,'bits':8,'mode':'affine'}
(target/'config.json').write_text(json.dumps(config))
