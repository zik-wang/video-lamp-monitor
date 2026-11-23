"""Lamp-relative luminous-center inference; never equate upper with yellow by default."""
import json,numpy as np
from pathlib import Path
LABELS={'green':'绿色','red':'红色','yellow':'黄色','green_position':'下层绿（位置推断）','upper_unknown':'上层亮（红黄待确认）','unresolved':'颜色未确认','off':'暗','unknown':'无法判断'}
def build_color_map(model,profile,position_rule=True):
 profile=Path(profile);meta=json.loads((profile/'profile.json').read_text(encoding='utf-8')) if (profile/'profile.json').exists() else {};overrides=meta.get('color_overrides',{});out={};evidence=[]
 for j,p in enumerate(model.cfg):
  img=model.centers[j].reshape(3,8,6,3);hi=img.max(axis=-1);lo=img.min(axis=-1);white=(lo>.80)&((hi-lo)<.35)
  # Weight central pixels so a nearby bright screen does not dominate a lamp.
  wx=np.exp(-((np.arange(6)-2.5)/1.8)**2)[None,:];ys=[]
  for k in range(3):
   weights=white[k]*wx;mass=weights.sum();ys.append(float((weights*np.arange(8)[:,None]/7).sum()/mass) if mass>=.7 else None)
  active=[k for k,s in enumerate(model.labels[j]) if s=='on' and ys[k] is not None];positions=[ys[k] for k in active];spread=max(positions)-min(positions) if len(positions)>1 else 0
  usable=spread>=.18 and spread*(p['roi'][3]-p['roi'][1])>=1.5
  for k in range(3):
   key=p['id']+f'_{k}';state=model.labels[j][k];color='off' if state=='off' else ('unknown' if state=='unknown' else 'unresolved');basis='未完成颜色校准'
   if position_rule and state=='on' and usable and ys[k] is not None:
    color='green_position' if ys[k]>=max(positions)-.08 else 'upper_unknown';basis='同一灯位发光中心上下分离，按本批下绿上红黄规则推断'
   if key in overrides:color=overrides[key];basis=meta.get('override_basis','用户确认颜色映射')
   out[key]=color;evidence.append(dict(key=key,color=color,label=LABELS[color],center_y=ys[k],position_separation=spread,basis=basis))
 return out,evidence
