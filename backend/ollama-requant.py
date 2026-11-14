import argparse,os,sys,signal,subprocess,json,tempfile,shutil,re
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--model',required=True);p.add_argument('--name',required=True);p.add_argument('--bits',type=int,choices=[4,8],required=True);p.add_argument('--worker',action='store_true')
a=p.parse_args()
if not a.worker:
 if os.getpgrp()!=os.getpid(): os.setsid()
 signal.signal(signal.SIGTERM,lambda s,f:sys.exit(130))
 sys.exit(subprocess.call(['uv','run','--no-project','--python','3.13','--with','mlx','python',__file__,'--worker','--model',a.model,'--name',a.name,'--bits',str(a.bits)]))
import mlx.core as mx
import importlib.util
spec=importlib.util.spec_from_file_location('helper',Path(__file__).with_name('import-model.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
try:
 if not re.fullmatch(r'[a-z0-9][a-z0-9._-]*(?::[a-z0-9][a-z0-9._-]*)?',a.name):raise ValueError('Недопустимое новое имя')
 destination=a.name if ':' in a.name else a.name+':latest'
 models=h.api('/api/tags')['models']
 original=next(m for m in models if m['name']==a.model)
 if any(m['name']==destination for m in models):raise ValueError('Имя копии уже занято')
 root=Path(os.environ.get('OLLAMA_MODELS',str(Path.home()/'.ollama/models')))
 # Select the manifest by its content digest, avoiding model-name path assumptions.
 import hashlib
 manifests=[]
 for path in (root/'manifests').rglob('*'):
  if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==original['digest'].removeprefix('sha256:'):
   manifests.append(path)
 if not manifests:
  model_path,tag=a.model.rsplit(':',1)
  parts=model_path.split('/')
  if len(parts)==1:parts=['registry.ollama.ai','library']+parts
  elif len(parts)==2:parts=['registry.ollama.ai']+parts
  candidate=root/'manifests'/Path(*parts)/tag
  if candidate.is_file() and candidate.resolve().is_relative_to((root/'manifests').resolve()):manifests=[candidate]
 if not manifests:raise ValueError('Локальный манифест не найден. Проверь OLLAMA_MODELS.')
 manifest=json.loads(manifests[0].read_text())
 layers=manifest['layers']
 if not any(x['mediaType']=='application/vnd.ollama.image.tensor' for x in layers):
  raise ValueError('Эта модель хранится в GGUF. В этой версии доступно повторное квантование установленных MLX-моделей.')
 def blob(layer):
  digest=layer['digest']
  if not re.fullmatch(r'sha256:[0-9a-f]{64}',digest):raise ValueError('Некорректный digest')
  return root/'blobs'/digest.replace(':','-')
 with tempfile.TemporaryDirectory(prefix='ollama-requant-') as temp:
  folder=Path(temp)/'weights';folder.mkdir()
  tensors_layers=[l for l in layers if l['mediaType']=='application/vnd.ollama.image.tensor' and not l.get('name','').startswith(('draft/','draft.'))]
  weight_map={}
  for index,layer in enumerate(tensors_layers):
   tensors,meta=mx.load(str(blob(layer)),format="safetensors",return_metadata=True)
   output={}
   for key,weight in tensors.items():
    if any(key.endswith(suffix) and key[:-len(suffix)] in tensors for suffix in ('.scale','.bias','.global_scale')):continue
    scale=tensors.get(key+'.scale')
    if scale is not None:
     mode=meta.get(key+'.quant_type',meta.get('quant_type',''))
     if mode.startswith('int'):
      bits=int(mode[3:]);group=weight.shape[-1]*32//bits//scale.shape[-1]
      dense=mx.dequantize(weight,scale,tensors.get(key+'.bias'),group_size=group,bits=bits,mode='affine')
     elif mode in ('mxfp4','mxfp8','nvfp4'):
      dense=mx.dequantize(weight,scale,mode=mode,global_scale=tensors.get(key+'.global_scale'))
     else:raise ValueError('Неизвестный формат весов: '+mode)
     # Write dense tensors; Ollama performs its own supported quantization on import.
     output[key]=dense
    else:output[key]=weight
   filename=f'model-{index+1:05}-of-{len(tensors_layers):05}.safetensors'
   mx.save_safetensors(str(folder/filename),output)
   weight_map.update({key:filename for key in output})
   del tensors,output
   mx.clear_cache()
   if index%50==0:print(f'Восстановление весов {index+1}/{len(tensors_layers)}',flush=True)
  for layer in layers:
   if layer['mediaType']=='application/vnd.ollama.image.json':
    name=layer.get('name','')
    if name.startswith('draft/'):continue
    if Path(name).name!=name:raise ValueError('Недопустимое имя конфигурации')
    shutil.copy2(blob(layer),folder/name)
  (folder/'model.safetensors.index.json').write_text(json.dumps({'weight_map':weight_map}))
  config=json.loads((folder/'config.json').read_text())
  for key in ('quantization','quantization_config','compression_config'):config.pop(key,None)
  (folder/'config.json').write_text(json.dumps(config))
  modelfile=Path(temp)/'Modelfile';modelfile.write_text('FROM ./weights\n')
  print('Квантование и импорт копии…',flush=True)
  subprocess.run(['ollama','create',destination,'-f',str(modelfile),'-q',f'int{a.bits}'],check=True)
  result=h.api('/api/generate',{'model':destination,'prompt':'2+2=','stream':False,'keep_alive':0,'options':{'num_ctx':4096,'num_predict':16}},timeout=180)
  if not result.get('done'):raise ValueError('Проверка запуска не завершена')
  print('Копия создана и запускается: '+destination,flush=True)
except Exception as e:sys.exit('Ошибка: '+str(e))
