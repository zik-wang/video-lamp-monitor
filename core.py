import cv2,numpy as np,json,time
from pathlib import Path
class Aligner:
    def __init__(self,reference):
        self.reference=reference;self.h,self.w=reference.shape[:2]
        self.orb=cv2.ORB_create(nfeatures=1400,fastThreshold=12)
        self.matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
        self.mask=np.full((self.h,self.w),255,np.uint8);self.mask[:65]=0
        self.key,self.desc=self.orb.detectAndCompute(cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY),self.mask)
    def align(self,frame):
        if frame.shape[:2]!=(self.h,self.w):return None,None,dict(reason='size_changed')
        key,desc=self.orb.detectAndCompute(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),self.mask)
        if desc is None or self.desc is None:return None,None,dict(reason='no_features')
        matches=[p[0] for p in self.matcher.knnMatch(self.desc,desc,k=2) if len(p)==2 and p[0].distance<.72*p[1].distance]
        if len(matches)<16:return None,None,dict(reason='few_matches',matches=len(matches))
        src=np.float32([self.key[m.queryIdx].pt for m in matches]);dst=np.float32([key[m.trainIdx].pt for m in matches])
        M,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=1500)
        if M is None:return None,None,dict(reason='registration_failed')
        good=inliers.ravel().astype(bool);n=int(good.sum());ratio=n/len(matches)
        scale=float(np.linalg.norm(M[:,0]));angle=float(np.degrees(np.arctan2(M[1,0],M[0,0])))
        spread=np.ptp(src[good],axis=0) if n else [0,0]
        if n<15 or ratio<.4 or not .6<scale<1.6 or abs(angle)>25 or spread[0]<100 or spread[1]<70:
            return None,None,dict(reason='registration_untrusted',inliers=n,ratio=ratio)
        aligned=cv2.warpAffine(frame,M,(self.w,self.h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        return aligned,M,dict(reason='ok',inliers=n,ratio=ratio,scale=scale,angle=angle)

class AlarmMachine:
    """Initial anomaly notification after 0.15s; separate type updates; unknown never clears an alarm."""
    def __init__(self,steady=3.,end_gap=4.):
        self.steady=steady;self.end_gap=end_gap;self.last=None;self.first=None;self.last_on=None
        self.run_start=None;self.off_start=None;self.active=False;self.mode=None;self.pulses=[];self.unknown_since=None;self.unknown_sent=False
    def reset_episode(self):
        self.first=None;self.last_on=None;self.run_start=None;self.off_start=None;self.active=False;self.mode=None;self.pulses=[]
    def step(self,t,state):
        out=[]
        def emit(kind,onset):out.append(dict(event=kind,onset=onset,detected_at=t))
        if state=='unknown':
            if self.unknown_since is None:self.unknown_since=t
            if not self.unknown_sent and t-self.unknown_since>=.15:
                emit('monitor_unavailable',self.unknown_since);self.unknown_sent=True
            self.last='unknown';self.run_start=None;return out
        if self.last=='unknown':
            # Start a new observed episode. Do not bridge an unseen time gap.
            if self.unknown_sent:emit('visibility_restored',t)
            self.reset_episode();self.unknown_since=None;self.unknown_sent=False
        if state=='on':
            if self.first is None:self.first=t
            if self.last!='on':self.run_start=t;self.pulses.append(t)
            self.last_on=t;self.off_start=None
            if not self.active and t-self.first>=.15:emit('alarm_on',self.first);self.active=True
            if self.run_start is not None and t-self.run_start>=self.steady and self.mode!='steady':
                emit('to_steady' if self.mode=='blink' else 'steady',self.run_start);self.mode='steady'
            elif len(self.pulses)>=2 and self.mode!='blink' and self.last!='on':
                emit('blinking',self.pulses[-2]);self.mode='blink'
        else:
            if self.last!='off':self.off_start=t
            if self.active and self.off_start is not None and t-self.off_start>=self.end_gap:
                emit('alarm_ended',self.off_start);self.reset_episode()
            elif not self.active and self.first is not None and self.last_on is not None and t-self.last_on>=self.end_gap:
                self.reset_episode()
        self.last=state;return out
class Debouncer:
 def __init__(self,seconds=.15):self.seconds=seconds;self.current='unknown';self.pending=None;self.since=None
 def step(self,t,value):
  if value==self.current:self.pending=None;return self.current,None
  if value!=self.pending:self.pending=value;self.since=t
  if t-self.since+1e-9>=self.seconds:
   self.current=value;self.pending=None;return self.current,self.since
  return self.current,None

class AppearanceModel:
 def __init__(self,folder):
  self.folder=Path(folder);self.cfg=json.loads((self.folder/'inventory.json').read_text(encoding='utf-8-sig'))['lamps'];m=np.load(self.folder/'appearance-model.npz');self.centers=m['centers'];self.radii=m['radii'];self.guards=m['guards'].astype(np.float32)/255
  labels={p['key']:p for p in json.loads((self.folder/'appearance-labels.json').read_text(encoding='utf-8-sig'))};labels.update({p['key']:p for p in json.loads((self.folder/'appearance-labels-recheck.json').read_text(encoding='utf-8-sig'))});self.min_guard_std=self.guards.mean(axis=-1).std(axis=(2,3)).min(axis=1);self.labels=[];samples={p['key']:p for p in json.loads((self.folder/'exemplars.json').read_text(encoding='utf-8-sig'))};self.conflicts=[]
  for p in self.cfg:
   row=[labels[p['id']+f'_{k}'] for k in range(3)];scores=[]
   for k in range(3):
    im=np.array(samples[p['id']+f'_{k}']['crop'],np.uint8);hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);scores.append(float(((hsv[:,:,2]>220)&(hsv[:,:,1]<100)).mean()))
   offs=[scores[k] for k,r in enumerate(row) if r['lit']=='off'];ons=[scores[k] for k,r in enumerate(row) if r['lit']=='on']
   states=[]
   for k,r in enumerate(row):
    state=r['lit'] if r['visible'] and r['confidence']!='low' else 'unknown'
    # A purported off appearance intermediate between dark and luminous samples is not safe to call off.
    if state=='off' and len(offs)>=2 and ons and scores[k]>.035 and scores[k]>min(offs)+.025 and scores[k]>=.35*max(ons):
     state='unknown';self.conflicts.append(dict(key=r['key'],reason='model_off_conflicts_with_luminous_core',white_score=scores[k]))
    states.append(state)
   self.labels.append(states)
  meta=self.folder/'profile.json'
  overrides=json.loads(meta.read_text(encoding='utf-8')).get('color_overrides',{}) if meta.exists() else {}
  for j,p in enumerate(self.cfg):
   for k in range(3):
    key=p['id']+f'_{k}'
    if key in overrides:self.labels[j][k]='off' if overrides[key]=='off' else ('unknown' if overrides[key]=='unknown' else 'on')
 def classify(self,cores,guards,valid):
  X=np.asarray(cores,np.float32).reshape(len(self.cfg),-1)/255;dist=((X[:,None,:]-self.centers)**2).mean(axis=2);k=dist.argmin(axis=1);j=np.arange(len(k));core_err=dist[j,k];g=np.asarray(guards,np.float32)/255;guard_err=np.abs(g-self.guards[j,k]).mean(axis=(1,2,3));structure=g.mean(axis=-1).std(axis=(1,2));good=np.asarray(valid)&(core_err<=self.radii[j,k])&(guard_err<.22)&(structure>=np.maximum(.012,.3*self.min_guard_std))
  states=[self.labels[i][int(kk)] if good[i] else 'unknown' for i,kk in enumerate(k)]
  return states,k,core_err,guard_err

def to_runs(states,ts,end):
 out=[]
 for state,t in zip(states,ts):
  if not out or out[-1]['state']!=state:
   if out:out[-1]['end']=float(t)
   out.append(dict(start=float(t),end=end,state=state))
 return out
def extract(im,M,cfg):
 H,W=im.shape[:2];cores=[];guards=[];valid=[]
 for p in cfg:
  a,b,c,d=p['roi'];ax=max(0,a-6);by=max(0,b-6);cx=min(W,c+6);dy=min(H,d+6)
  if M is None:crop=im[by:dy,ax:cx];inside=False
  else:
   local=M.copy();local[:,2]=M@np.array([ax,by,1.]);crop=cv2.warpAffine(im,local,(cx-ax,dy-by),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP);corners=cv2.transform(np.float32([[[a,b],[c,b],[c,d],[a,d]]]),M)[0];inside=bool(((corners[:,0]>=0)&(corners[:,0]<W)&(corners[:,1]>=0)&(corners[:,1]<H)).all())
  cores.append(cv2.resize(crop[b-by:d-by,a-ax:c-ax],(6,8),interpolation=cv2.INTER_AREA));guards.append(cv2.resize(crop,(6,8),interpolation=cv2.INTER_AREA));valid.append(inside)
 return np.array(cores),np.array(guards),valid
def typed_intervals(runs, steady=3.0, end_gap=4.0):
    if not runs:return []
    for i,r in enumerate(runs):
        if r['state'] not in ('on','off','unknown') or r['end']<=r['start']:
            raise ValueError('Invalid input run')
        if i and abs(r['start']-runs[i-1]['end'])>1e-6:
            raise ValueError('Runs must be contiguous and ordered')
        if i and r['state']==runs[i-1]['state']:
            raise ValueError('Merge adjacent identical runs first')
    start,end=runs[0]['start'],runs[-1]['end']
    out=[]; group=[]
    def add(kind,a,b,pulses=0,left=False,right=False,reason='state_change'):
        if b-a<1e-8:return
        on=sum(max(0,min(b,r['end'])-max(a,r['start'])) for r in runs if r['state']=='on')
        out.append(dict(type=kind,start=a,end=b,duration=b-a,on_seconds=on,off_seconds=(b-a-on if kind!='unknown' else None),pulses=pulses,left_censored=left,right_censored=right,end_reason=reason))
    def flush(reason='confirmed_off'):
        nonlocal group
        if not group:return
        i=0
        while i<len(group):
            r=group[i];j=i+1
            if r['end']-r['start']>=steady-1e-8:
                kind='on'
            else:
                while j<len(group) and group[j]['end']-group[j]['start']<steady-1e-8:j+=1
                kind='blink' if j-i>=2 else 'short'
            b=group[j]['start'] if kind=='blink' and j<len(group) else group[j-1]['end']
            add(kind,r['start'],b,j-i,left=abs(r['start']-start)<1e-8,right=(j==len(group) and reason in ('video_end','unknown')),reason=reason if j==len(group) else 'type_change')
            i=j
        group=[]
    for r in runs:
        if r['state']=='on':group.append(r)
        elif r['state']=='unknown':
            flush('unknown');add('unknown',r['start'],r['end'],left=r['start']==start,right=r['end']==end,reason='visibility_unknown')
        elif r['end']-r['start']>=end_gap-1e-8:flush()
    flush('video_end')
    out.sort(key=lambda r:r['start'])
    # Complete the observed timeline without treating unknown as dark.
    full=[];cursor=start
    for r in out:
        if r['start']>cursor+1e-8:
            full.append(dict(type='off',start=cursor,end=r['start'],duration=r['start']-cursor,on_seconds=0.,off_seconds=r['start']-cursor,pulses=0,left_censored=cursor==start,right_censored=False,end_reason='state_change'))
        full.append(r);cursor=r['end']
    if cursor<end-1e-8:
        full.append(dict(type='off',start=cursor,end=end,duration=end-cursor,on_seconds=0.,off_seconds=end-cursor,pulses=0,left_censored=cursor==start,right_censored=True,end_reason='video_end'))
    assert abs(sum(r['duration'] for r in full)-(end-start))<1e-6
    for a,b in zip(full,full[1:]):assert abs(a['end']-b['start'])<1e-6
    return full
class CausalLampMachine(AlarmMachine):
 """A long steady run must not become the first pulse of a later blink episode."""
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.long_run_ended=False
 def step(self,t,state):
  if state=='off' and self.last=='on' and self.run_start is not None and t-self.run_start>=self.steady:
   self.pulses=[];self.long_run_ended=True
  if state=='on' and self.last=='off' and self.long_run_ended:
   self.mode=None;self.long_run_ended=False
  if state=='unknown':self.long_run_ended=False
  return super().step(t,state)
