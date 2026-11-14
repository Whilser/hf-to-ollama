import json,sys,urllib.request,urllib.error,time,statistics,signal
def api(path,data=None,method=None):
 req=urllib.request.Request('http://127.0.0.1:11434/api/'+path,data=json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/json'},method=method)
 with urllib.request.urlopen(req,timeout=180) as r:
  raw=r.read()
  return json.loads(raw) if raw else {}
def emit(event,**data): print(json.dumps(dict(event=event,**data)),flush=True)
def models(): return api('tags')['models']
def stop(signum, frame):
 raise SystemExit(130)
signal.signal(signal.SIGTERM, stop)
try:
 action=sys.argv[1]
 if action=='list': emit('models',models=models())
 elif action=='delete':
  api('delete',{'model':sys.argv[2]},'DELETE')
  emit('message',text='Модель удалена')
 elif action=='rename':
  old,new=sys.argv[2:4]
  if ':' not in new: new+=':latest'
  before=models()
  if any(m['name']==new for m in before): raise ValueError('Это имя уже занято')
  original=next(m for m in before if m['name']==old)
  api('copy',{'source':old,'destination':new})
  copied=next(m for m in models() if m['name']==new)
  if copied['digest']!=original['digest']: raise ValueError('Проверка копии не прошла. Исходная модель сохранена.')
  api('delete',{'model':old},'DELETE')
  emit('message',text='Модель переименована')
 elif action=='test':
  items=models();emit('models',models=items)
  for i,item in enumerate(items):
   name=item['name'];emit('message',text=f'Тест {i+1}/{len(items)} · {name}')
   try:
    api('generate',{'model':name,'keep_alive':0})
    deadline=time.monotonic()+30
    while any(m['name']==name for m in api('ps')['models']):
     if time.monotonic()>deadline: raise RuntimeError('Не удалось выгрузить модель перед тестом')
     time.sleep(.5)
    show=api('show',{'model':name})
    if 'completion' not in show.get('capabilities',[]): raise RuntimeError('Тест предназначен для текстовых моделей')
    payload={'model':name,'messages':[{'role':'user','content':'Explain how a computer stores and retrieves information. Give a detailed answer.'}],'stream':False,'keep_alive':'5m','options':{'num_ctx':4096,'num_predict':128,'temperature':0,'seed':42}}
    if 'thinking' in show.get('capabilities',[]): payload['think']=False
    result=api('chat',payload)
    loaded=next((m for m in api('ps')['models'] if m['name']==name),None)
    rates=[]
    for run in range(2):
     r=result if run==0 else api('chat',payload)
     if r.get('eval_count',0) and r.get('eval_duration',0): rates.append(r['eval_count']/(r['eval_duration']/1e9))
    if not rates: raise RuntimeError('Ollama не вернула скорость генерации')
    emit('result',name=name,load=result.get('load_duration',0)/1e9,memory=loaded['size']/1e9 if loaded else None,speed=statistics.mean(rates))
   except Exception as e: emit('error',text=name+' · '+str(e))
   finally:
    try: api('generate',{'model':name,'keep_alive':0})
    except Exception: pass
  emit('message',text='Тестирование завершено')
except Exception as e:
 emit('error',text=str(e));sys.exit(1)
