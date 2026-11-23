"""Bounded model client: structured outputs, saved evidence, retry and cancellation."""
import json,time,base64,urllib.request,cv2
from pathlib import Path
class Cancelled(RuntimeError):pass
class ModelClient:
 def __init__(self,endpoint,model,out,cancel,budget=18):self.endpoint=endpoint.rstrip('/');self.model=model;self.out=Path(out);self.cancel=cancel;self.budget=budget;self.calls=0
 def request(self,prompt,images=(),validate=None):
  last=None
  for attempt in range(2):
   if self.cancel.is_set():raise Cancelled('已停止')
   if self.calls>=self.budget:raise RuntimeError('已达到本次模型请求上限，保留现有证据，请缩小范围或检查模型输出。')
   self.calls+=1;content=[dict(type='text',text=prompt if attempt==0 else prompt+'\nPrevious output failed validation: '+str(last)+'. Correct only the JSON structure/constraints.')]
   for im in images:
    data=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,92])[1].tobytes();content.append(dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(data).decode())))
   body=dict(model=self.model,messages=[dict(role='user',content=content)],temperature=0,max_tokens=3200,chat_template_kwargs=dict(enable_thinking=False));req=urllib.request.Request(self.endpoint+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'});start=time.monotonic()
   try:
    with urllib.request.urlopen(req,timeout=60) as r:raw=json.load(r)
    (self.out/f'model-{self.calls:02}.json').write_text(json.dumps(dict(prompt=prompt,elapsed=time.monotonic()-start,response=raw),ensure_ascii=False,indent=2),encoding='utf-8')
    text=raw['choices'][0]['message']['content'].strip()
    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
    obj=json.loads(text)
    if validate:validate(obj)
    if self.cancel.is_set():raise Cancelled('已停止')
    return obj
   except Cancelled:raise
   except Exception as e:last=e
  raise RuntimeError('视觉模型未返回可用结果：'+str(last))
