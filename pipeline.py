"""Fixed harness: inspect -> locate/deduplicate -> calibrate -> scan -> report."""
import json,time
from datetime import datetime,timedelta
from pathlib import Path
import cv2,numpy as np
from core import Aligner,AppearanceModel,Debouncer,extract,typed_intervals,to_runs
from colors import LABELS
from vision import Cancelled
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def check(cancel):
 if cancel.is_set():raise Cancelled('任务已停止，已有文件保留')
def rectangles(w,h):
 # Whole-frame coverage, including lower rows; overlapping crops preserve boundary objects.
 for iy in range(2):
  for ix in range(3):
   yield [max(0,int(ix*w/3-w*.045)),max(0,int(iy*h/2-h*.04)),min(w,int((ix+1)*w/3+w*.045)),min(h,int((iy+1)*h/2+h*.04))]
def locate(ref,client,notify,cancel):
 h,w=ref.shape[:2];records=[]
 def valid(o):
  assert isinstance(o.get('lamps'),list)
  for p in o['lamps']:
   b=p.get('head_box');assert isinstance(b,list) and len(b)==4 and all(isinstance(v,(int,float)) for v in b);assert 0<=b[0]<b[2]<=1000 and 0<=b[1]<b[3]<=1000
 for i,(a,b,c,d) in enumerate(rectangles(w,h)):
  check(cancel);notify('定位灯位',f'Qwen检查第{i+1}/6个画面分块',.05+i*.025)
  crop=ref[b:d,a:c];scale=min(3,1200/max(crop.shape[:2]));crop=cv2.resize(crop,None,fx=scale,fy=scale)
  prompt='找出这张工厂局部图中所有机器塔式信号灯，包括远处小灯和未亮灯。灯头在细长白/灰灯杆顶部；有些近处灯杆很短。不要把屏幕、标牌、反光、管道当灯。一个多色灯柱算一盏。返回JSON {"lamps":[{"head_box":[x1,y1,x2,y2],"confidence":"high|medium|low"}]}，坐标0到1000相对于整张局部图。框紧贴整个灯头，不包括长杆。不要预设数量。'
  result=client.request(prompt,[crop],valid)
  for p in result['lamps']:
   box=[a+p['head_box'][0]/1000*(c-a),b+p['head_box'][1]/1000*(d-b),a+p['head_box'][2]/1000*(c-a),b+p['head_box'][3]/1000*(d-b)]
   if box[2]-box[0]>w*.08 or box[3]-box[1]>h*.25:continue
   records.append(dict(source_id=f'R{len(records)+1:03}',bbox=box,center=[(box[0]+box[2])/2,(box[1]+box[3])/2]))
 # A second scale is chosen from small detected heads, with extra space toward the vanishing point.
 # This uses no camera-specific coordinates and no target lamp count.
 coarse=list(records)
 for col in range(3):
  aa=max(0,int(col*w/3-w*.045));cc=min(w,int((col+1)*w/3+w*.045));small=[p for p in coarse if aa<=p['center'][0]<=cc and p['bbox'][3]-p['bbox'][1]<h*.045]
  if not small:continue
  bb=max(0,int(min(p['bbox'][1] for p in small)-h*.055));dd=min(h,int(max(p['bbox'][3] for p in small)+h*.045))
  crop=ref[bb:dd,aa:cc];crop=cv2.resize(crop,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
  notify('补查远处小灯',f'Qwen放大检查第{col+1}列灯头区域',.20+col*.015)
  result=client.request('请逐个检查这张放大的工厂局部图，找全远处和近处的塔式信号灯。一根独立白色细杆顶部的彩色灯头是一盏，紧邻的不同灯杆不能漏掉或合并；灯暗着也算。框住完整灯头，不把白色长杆框进去。屏幕、墙面电线、标识不是灯。严格JSON {"lamps":[{"head_box":[x1,y1,x2,y2],"confidence":"high|medium|low"}]}。坐标0到1000相对于整张局部图。不要预设个数。',[crop],valid)
  for p in result['lamps']:
   box=[aa+p['head_box'][0]/1000*(cc-aa),bb+p['head_box'][1]/1000*(dd-bb),aa+p['head_box'][2]/1000*(cc-aa),bb+p['head_box'][3]/1000*(dd-bb)]
   if box[2]-box[0]>w*.08 or box[3]-box[1]>h*.25:continue
   records.append(dict(source_id=f'R{len(records)+1:03}',bbox=box,center=[(box[0]+box[2])/2,(box[1]+box[3])/2]))
 if not records:raise RuntimeError('模型没有找到可用灯位；已保存分块识别记录。')
 save(client.out/'location-records.json',records)
 def groups_valid(o):
  groups=o['groups'];ids=[x for g in groups for x in g['members']];assert sorted(ids)==sorted(p['source_id'] for p in records) and len(ids)==len(set(ids))
  lookup={p['source_id']:p for p in records}
  for g in groups:
   assert g['representative'] in g['members'];pp=[lookup[x] for x in g['members']]
   if len(pp)>1:
    xs=[p['center'][0] for p in pp];ys=[p['center'][1] for p in pp];width=max(p['bbox'][2]-p['bbox'][0] for p in pp);height=max(p['bbox'][3]-p['bbox'][1] for p in pp)
    assert max(xs)-min(xs)<=max(3,.65*width) and max(ys)-min(ys)<=max(6,.75*height),'相邻不同灯位被合并'
 prompt='仅合并同一灯头的重复坐标记录。相邻不同灯杆上的灯不能合并，不删除任何记录；没有重复的记录单独成组。每个source_id恰好出现一次。输出JSON {"groups":[{"members":["R001"],"representative":"R001"}]}。选择完整灯头的最紧致框作代表，优先不要包含长白色杆的框，不创造坐标。记录：'+json.dumps(records)
 result=client.request(prompt,validate=groups_valid);lookup={p['source_id']:p for p in records};heads=[lookup[g['representative']] for g in result['groups']];heads.sort(key=lambda p:p['center'][0]);cfg=[]
 for i,p in enumerate(heads,1):
  a,b,c,d=p['bbox'];cfg.append(dict(id=f'L{i:02}',roi=[max(0,int(a)-1),max(0,int(b)-1),min(w,int(np.ceil(c))+1),min(h,int(np.ceil(d))+1)],center=p['center'],alias=''))
 if len(cfg)>60:raise RuntimeError('候选超过60盏，可能混入大量误检；请先检查灯位图与模型响应。')
 return cfg

def scan(video,ref,cfg,cancel,notify,limit=None):
 cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));expected=min(expected,round(limit*fps)) if limit else expected;aligner=Aligner(ref);M=None;cores=[];guards=[];valid=[];ts=[];n=0;t0=time.monotonic();qualities=[]
 while n<expected:
  check(cancel);ok,im=cap.read()
  if not ok:break
  if n%max(1,int(fps))==0:_,M,q=aligner.align(im);qualities.append(q['reason'])
  a,b,c=extract(im,M,cfg);cores.append(a);guards.append(b);valid.append(c);ts.append(n/fps);n+=1
  if n%max(1,int(fps*10))==0:notify('本地逐帧检测',f'{n}/{expected} 帧',.3+.42*n/expected)
 cap.release()
 if n<expected-1:raise RuntimeError(f'视频解码提前结束：{n}/{expected}帧，不生成完整报告。')
 return dict(cores=np.asarray(cores,np.uint8),guards=np.asarray(guards,np.uint8),valid=np.asarray(valid,bool),times=np.array(ts),fps=float(fps),duration=n/fps,seconds=time.monotonic()-t0,registration_failures=sum(q!='ok' for q in qualities))

def calibrate(data,cfg,profile,client,notify,cancel,video,reference):
 arr=data['cores'];guards=data['guards'];valid=data['valid'];ts=data['times'];fps=data['fps'];centers=[];radii=[];templates=[];samples=[];cv2.setRNGSeed(19)
 for j,p in enumerate(cfg):
  check(cancel);idx=np.arange(0,len(ts),max(1,int(fps)));idx=idx[valid[idx,j]]
  if len(idx)<3:raise RuntimeError('可用校准帧不足，无法建立灯位外观模型。')
  X=arr[idx,j].reshape(len(idx),-1).astype(np.float32)/255;_,lab,C=cv2.kmeans(X,3,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,50,.0001),4,cv2.KMEANS_PP_CENTERS);centers.append(C);rr=[];gg=[]
  for k in range(3):
   mask=lab.ravel()==k;dist=((X[mask]-C[k])**2).mean(axis=1);member=idx[mask];chosen=int(member[np.argmin(dist)]);rr.append(max(.022,float(np.quantile(dist,.99))*2.5));gg.append(guards[chosen,j]);samples.append(dict(key=p['id']+f'_{k}',lamp=p['id'],cluster=k,frame=chosen,t=float(ts[chosen]),population=int(mask.sum()),crop=arr[chosen,j].tolist(),guard=guards[chosen,j].tolist()))
  radii.append(rr);templates.append(gg)
 np.savez_compressed(profile/'appearance-model.npz',centers=np.array(centers),radii=np.array(radii),guards=np.array(templates));save(profile/'exemplars.json',samples);labels=[]
 native={};cap=cv2.VideoCapture(str(video));aligner=Aligner(reference)
 for sample in sorted(samples,key=lambda s:s['frame']):
  check(cancel);cap.set(cv2.CAP_PROP_POS_FRAMES,sample['frame']);ok,frame=cap.read()
  if not ok:raise RuntimeError('校准样本无法解码')
  aligned,_,_=aligner.align(frame)
  if aligned is None:raise RuntimeError('校准样本无法配准')
  p=next(p for p in cfg if p['id']==sample['lamp']);a,b,c,d=p['roi'];patch=aligned[max(0,b-6):min(aligned.shape[0],d+6),max(0,a-6):min(aligned.shape[1],c+6)]
  native[sample['key']]=patch
 cap.release()
 for batch in range(0,len(samples),21):
  ss=samples[batch:batch+21];sheet=np.full((int(np.ceil(len(ss)/3))*170,540,3),240,np.uint8)
  for i,s in enumerate(ss):
   xx=i%3*180;yy=i//3*170;patch=native[s['key']];scale=min(150/patch.shape[1],130/patch.shape[0]);patch=cv2.resize(patch,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST);ph,pw=patch.shape[:2];sheet[yy+32:yy+32+ph,xx+(180-pw)//2:xx+(180-pw)//2+pw]=patch;cv2.putText(sheet,s['key'],(xx+8,yy+23),0,.55,(0,0,0),1)
  keys=[s['key'] for s in ss]
  def validate(o):
   assert len(o['items'])==len(keys) and {x['key'] for x in o['items']}==set(keys)
   assert all(x['lit'] in ['on','off','unknown'] and isinstance(x['visible'],bool) and x['confidence'] in ['high','medium','low'] for x in o['items'])
  notify('校准灯光状态',f'Qwen核对{batch+1}—{batch+len(ss)}号样本',.75)
  prompt='逐项判断标记的灯头样本，只看中央灯头，不看旁边屏幕。白亮或有明确彩色发光灯芯才是on；红黄绿塑料外壳本身不代表亮。遮挡、辨不出灯头用unknown。返回JSON {"items":[{"key":"L01_0","visible":true,"lit":"on|off|unknown","colors":["unresolved"],"confidence":"high|medium|low"}]}。只判亮暗，不猜颜色。不从单张图推断闪烁。必须包含且仅包含这些key：'+json.dumps(keys)
  labels+=client.request(prompt,[sheet],validate)['items']
 save(profile/'appearance-labels.json',labels);save(profile/'appearance-labels-recheck.json',[])

def annotated(ref,cfg,path):
 h,w=ref.shape[:2];im=cv2.resize(ref,None,fx=2,fy=2);used=[]
 for p in cfg:
  x,y=[round(v*2) for v in p['center']];col=(80,235,70);cv2.circle(im,(x,y),7,col,2)
  for dx,dy in [(-15,-22),(-15,-45),(0,30),(10,-70),(-40,55),(20,75),(-10,-95)]:
   lx=max(0,min(w*2-65,x+dx));ly=max(18,min(h*2-5,y+dy));box=(lx,ly-18,lx+62,ly+3)
   if not any(not(box[2]<q[0] or box[0]>q[2] or box[3]<q[1] or box[1]>q[3]) for q in used):break
  used.append(box);cv2.line(im,(x,y),(lx+15,ly),col,1);cv2.rectangle(im,(box[0],box[1]),(box[2],box[3]),(20,30,20),-1);cv2.putText(im,p['id'],(lx+2,ly-2),0,.55,col,1)
 cv2.imencode('.jpg',im)[1].tofile(str(path))

def summarize(data,model,color_map,settings,cancel):
 ts=data['times'];end=data['duration'];N=len(ts);count=len(model.cfg);states=np.empty((N,count),dtype='U12');colors=np.empty((N,count),dtype='U20');filters=[Debouncer() for _ in model.cfg];colorfilters=[Debouncer() for _ in model.cfg]
 for i,t in enumerate(ts):
  if i%300==0:check(cancel)
  ss,kk,_,_=model.classify(data['cores'][i],data['guards'][i],data['valid'][i])
  for j,s in enumerate(ss):
   key=model.cfg[j]['id']+f'_{int(kk[j])}';color=color_map[key] if s=='on' else s
   if color=='off':s='off'
   if color=='unknown':s='unknown'
   stable,since=filters[j].step(float(t),s);states[i,j]=stable
   if since is not None:states[np.searchsorted(ts,since):i+1,j]=stable
   cc,csince=colorfilters[j].step(float(t),color);colors[i,j]=cc
   if csince is not None:colors[np.searchsorted(ts,csince):i+1,j]=cc
 intervals=[];changes=[];alerts=[];summary=[];events=[];threshold=float(settings.get('alert_seconds',60));clock=settings.get('video_start','').strip();clock=datetime.fromisoformat(clock) if clock else None
 def wall(t):return (clock+timedelta(seconds=float(t))).isoformat(sep=' ',timespec='seconds') if clock else ''
 for j,p in enumerate(model.cfg):
  runs=to_runs(states[:,j],ts,end);spans=typed_intervals(runs,3.,4.);cruns=to_runs(colors[:,j],ts,end)
  for r in cruns:changes.append(dict(lamp=p['id'],start=r['start'],end=r['end'],state=r['state'],label=LABELS.get(r['state'],r['state']),wall=wall(r['start'])))
  for r in spans:
   kind={'on':'常亮','blink':'闪烁','short':'短亮','off':'熄灭','unknown':'无法判断'}[r['type']];lo=np.searchsorted(ts,r['start']);hi=np.searchsorted(ts,r['end']);seen=sorted(set(colors[lo:hi,j])-{'off','unknown'});color='、'.join(LABELS.get(x,x) for x in seen) or '未确认';desc=f'{p["id"]}：{kind}'+(f'，{color}' if r['type'] in ['on','blink','short'] else '')
   intervals.append(dict(lamp=p['id'],**r,kind=kind,color=color,description=desc));events.append(dict(lamp=p['id'],position=r['start'],wall=wall(r['start']),event=desc,source='外观校准 + 本地检测'))
   if r['type']=='blink' and r['duration']>=threshold:alerts.append(dict(lamp=p['id'],condition='闪烁持续达到门槛',start=r['start'],fired_at=r['start']+threshold,end=r['end'],duration=r['duration'],evidence='本地闪烁时段；含闪烁内部暗间隙'))
  # Yellow alerts need confirmed yellow, not an inferred upper red/yellow position.
  for r in cruns:
   if r['state']=='yellow' and r['end']-r['start']>=threshold:alerts.append(dict(lamp=p['id'],condition='黄色持续达到门槛',start=r['start'],fired_at=r['start']+threshold,end=r['end'],duration=r['end']-r['start'],evidence='已确认黄色映射'))
  summary.append(dict(lamp=p['id'],on_seconds=sum(r['end']-r['start'] for r in runs if r['state']=='on'),off_seconds=sum(r['end']-r['start'] for r in runs if r['state']=='off'),unknown_seconds=sum(r['end']-r['start'] for r in runs if r['state']=='unknown'),duration=end,blink_intervals=sum(r['type']=='blink' for r in spans)))
  assert abs(sum(r['duration'] for r in spans)-end)<1e-6
 # Color changes are separate signal records, so a green-to-red change is not lost during continuous light.
 last_color={}
 for r in changes:
  if r['state']=='unknown' or (r['state']=='off' and r['end']-r['start']>=4):last_color[r['lamp']]=None
  elif r['state']!='off':
   if last_color.get(r['lamp'])!=r['state'] and not any(e['lamp']==r['lamp'] and abs(e['position']-r['start'])<.001 for e in events):events.append(dict(lamp=r['lamp'],position=r['start'],wall=r['wall'],event='发光状态：'+r['label'],source='颜色校准 + 本地检测'))
   last_color[r['lamp']]=r['state']
 for seq in [intervals,changes,alerts]:seq.sort(key=lambda r:(r['lamp'],r['start']))
 events.sort(key=lambda r:(r['lamp'],r['position'],r['event']))
 return dict(intervals=intervals,color_intervals=changes,alerts=alerts,summary=summary,events=events,duration=end,frames=N,lamps=count,alert_seconds=threshold,clock_provided=bool(clock))
