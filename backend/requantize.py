import argparse,json,os,signal,subprocess,sys,tempfile,shutil
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--source',required=True);p.add_argument('--bits',type=int,choices=[4,8],required=True);p.add_argument('--name',required=True);p.add_argument('--worker',action='store_true')
a=p.parse_args()
if not a.worker:
 if os.getpgrp() != os.getpid(): os.setsid()
 signal.signal(signal.SIGTERM,lambda s,f:sys.exit(130))
 try:
  subprocess.run(['uv','run','--no-project','--python','3.13','--with','mlx','python',__file__,'--worker','--source',a.source,'--bits',str(a.bits),'--name',a.name],check=True)
 except subprocess.CalledProcessError as e:sys.exit(e.returncode)
 sys.exit()
import mlx.core as mx
import importlib.util
spec=importlib.util.spec_from_file_location('importer',Path(__file__).with_name('import-model.py'))
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
try:
 import re
 if not re.fullmatch(r'[a-z0-9][a-z0-9._-]*(?::[a-z0-9][a-z0-9._-]*)?',a.name):raise ValueError('Недопустимое имя модели')
 name=a.name if ':' in a.name else a.name+':latest'
 if any(m['name']==name for m in helper.api('/api/tags')['models']):raise ValueError('Имя уже занято')
 source=Path(a.source)
 config=json.loads((source/'config.json').read_text())
 quant=config.get('quantization') or config.get('quantization_config') or {}
 if quant.get('quant_method'):raise ValueError('Этот формат квантования пока не поддерживается')
 shards=list(source.glob('*.safetensors'))
 if not shards:raise ValueError('Выбери папку с Safetensors и config.json')
 with tempfile.TemporaryDirectory(prefix='hf-requant-') as temp:
  target=Path(temp)/'weights';target.mkdir()
  for index,path in enumerate(shards):
   print(f'Преобразование {index+1}/{len(shards)} · {path.name}',flush=True)
   tensors,metadata=mx.load(str(path),return_metadata=True)
   if any('.experts.switch_glu.' in key for key in tensors):raise ValueError('Эта раскладка MoE пока поддерживается только во вкладке импорта')
   count=0
   for key in list(tensors):
    if not key.endswith('.scales'):continue
    base=key[:-7];weight=base+'.weight';bias=base+'.biases'
    if weight not in tensors:continue
    settings=quant.get(base,quant)
    if not isinstance(settings,dict) or 'bits' not in settings:raise ValueError('Не удалось определить параметры исходного квантования')
    dense=mx.dequantize(tensors[weight],tensors[key],tensors.get(bias),group_size=settings.get('group_size',64),bits=settings['bits'],mode=settings.get('mode','affine'))
    tensors[weight],tensors[key],tensors[bias]=mx.quantize(dense,group_size=64,bits=a.bits,mode='affine')
    count+=1
   if count==0:raise ValueError('Для этой операции нужны уже квантованные MLX-веса')
   mx.save_safetensors(str(target/path.name),tensors,metadata)
   del tensors;mx.clear_cache()
  for path in source.iterdir():
   if path.is_file() and path.suffix!='.safetensors':shutil.copy2(path,target/path.name)
  for key in ('quantization','quantization_config'):config[key]={'bits':a.bits,'group_size':64,'mode':'affine'}
  (target/'config.json').write_text(json.dumps(config))
  modelfile=Path(temp)/'Modelfile';modelfile.write_text('FROM ./weights\n')
  subprocess.run(['ollama','create',name,'-f',str(modelfile)],check=True)
  print('Проверка запуска…',flush=True)
  result=helper.api('/api/generate',{'model':name,'prompt':'2+2=','stream':False,'keep_alive':0,'options':{'num_ctx':4096,'num_predict':16}},timeout=180)
  if not result.get('done'):raise ValueError('Проверка генерации не завершилась')
  print('Готово · '+name+'\nОтвет: '+result.get('response',''),flush=True)
except Exception as e:sys.exit('Ошибка: '+str(e))
