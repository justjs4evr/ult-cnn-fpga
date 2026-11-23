"""Keep all original CNN weights; train a full-frame crossing context head."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from inference import CLASSES,load_model
from finetune_crossing_steering import OUT,BASE,ORIGINAL,read,write,pixels
from finetune_baseline_crossings import key,dump
from crossing_context64_model import CrossingContext64CNN

def gray64(rows):
    return np.stack([np.asarray(Image.open(r['path']).convert('L').resize((64,64),Image.Resampling.BOX),dtype='float32')/255 for r in rows])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--augment',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(1264)
    run=OUT/('context64_augmented' if args.augment else 'context64_candidate');run.mkdir(exist_ok=False)
    old,ckpt=load_model(BASE);model=CrossingContext64CNN();model.base.load_state_dict(old.state_dict());model.base.requires_grad_(False)
    orig=read(ORIGINAL/'split_manifest.csv');crop=np.load(OUT/'preprocessing_cache.npz')['original'];new=read(OUT/'steering_capture/manifest.csv');nc=pixels(new,ckpt['preprocessing'])
    held={key(p) for p,r in zip(crop,orig) if r['split']!='train'}
    localheld={key(p) for p,r in zip(nc,new) if r['split']!='train'}
    with torch.no_grad():
        cx=torch.from_numpy(crop.astype('float32')[:,None]/255)
        opred=torch.cat([old(cx[i:i+512]).argmax(1) for i in range(0,len(cx),512)])
    selected=[];labels=[]
    # Head must leave historical turn decisions untouched. Existing forward
    # decisions are already identical motor actions to STOP and need no gate.
    for i,r in enumerate(orig):
        if int(opred[i]) not in (1,2,3,4) or r['source_class'] in ('crossroads','crosswalk'):continue
        if r['split']=='train' and key(crop[i]) in localheld:continue
        selected.append({**r,'source':'historical_turn','context_label':0});labels.append(0)
    for r,p in zip(new,nc):
        if r['split']=='train' and (key(p) in held or key(p) in localheld):continue
        label=int(abs(float(r['yaw_offset_deg']))<=5)
        selected.append({**r,'source':'controlled_crossing','context_label':label});labels.append(label)
    if args.augment:
        aug=read(OUT/'context_augmentation_capture/manifest.csv')
        selected.extend([{**r,'source':'controlled_crossing_augmentation','context_label':1} for r in aug]);labels.extend([1]*len(aug))
    # Dense successful corner trajectory: only actual turn commands need protection.
    roadrows=read(OUT/'dense_teacher_route/teacher/original_start.csv')
    roads=[]
    for p in sorted((OUT/'dense_teacher_route/teacher/original_start').glob('*.png')):
        j=int(p.stem.split('_')[0])-1
        if not args.augment:
            if roadrows[j]['predicted_class'] not in ('left','right','sharp_left','sharp_right'):continue
            label=0
        else:
            row=roadrows[j];x,y,h=float(row['x_px']),float(row['y_px']),float(row['heading_deg'])
            near_cross=1750<=x<=2350 and abs(y-1919)<=40
            aligned=abs((h+180)%360-180)<=5
            label=int(near_cross and aligned)
            if near_cross and not aligned:continue
        roads.append(dict(path=str(p),split='dev_route' if j%10==0 else 'train',source='successful_route_context',context_label=label))
    selected.extend(roads);labels.extend([r['context_label'] for r in roads])
    raw=gray64(selected)
    # This head sees 64x64 grayscale, so leakage is checked on that actual
    # input rather than the legacy CNN's lossy 16x16 crop.
    actual_keys=[key(np.rint(p*255).astype('uint8')) for p in raw]
    fullheld={k for k,r in zip(actual_keys,selected) if r['split']!='train'}
    seen=set();allowed=[];excluded=[]
    for i,(k,r) in enumerate(zip(actual_keys,selected)):
        if r['split']=='train' and (k in fullheld or k in seen):
            excluded.append({**r,'reason':'exact_or_mirrored_context_holdout_or_duplicate'});continue
        if r['split']=='train':seen.add(k)
        allowed.append(i)
    selected=[selected[i] for i in allowed];labels=[labels[i] for i in allowed];raw=raw[allowed]
    write(run/'excluded_inputs.csv',excluded)
    features=[]
    with torch.no_grad():
        for i in range(0,len(raw),256):features.append(model.context_features(torch.from_numpy(raw[i:i+256,None])))
    features=torch.cat(features);y=torch.tensor(labels,dtype=torch.float32)
    train=torch.tensor([i for i,r in enumerate(selected) if r['split']=='train']);dev=torch.tensor([i for i,r in enumerate(selected) if r['split'] in ('dev_frame','dev_local','dev_route')])
    pos=train[y[train]==1];neg=train[y[train]==0]
    opt=torch.optim.Adam(model.context_head.parameters(),lr=.001,weight_decay=.0001);history=[];best=-1;state=None;threshold=None
    for epoch in range(1,101):
        for _ in range(100):
            idx=torch.cat([pos[torch.randint(len(pos),(64,))],neg[torch.randint(len(neg),(192,))]])
            loss=F.binary_cross_entropy_with_logits(model.context_head(features[idx]).flatten(),y[idx])
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():dp=model.context_head(features[dev]).flatten().sigmoid()
        if args.augment:
            protected=torch.tensor([selected[int(i)]['source']!='historical_turn' for i in dev])&(y[dev]==0)
            historic=~torch.tensor([selected[int(i)]['source']!='historical_turn' for i in dev])&(y[dev]==0)
            cutoff=max(.8,float(dp[protected].max())+.001,float(torch.quantile(dp[historic],.9975))+.001)
        else:
            cutoff=max(.8,float(dp[y[dev]==0].max())+.001)
        recall=float((dp[y[dev]==1]>=cutoff).float().mean()) if cutoff<1 else 0.
        if recall>best:best=recall;state=copy.deepcopy(model.state_dict());threshold=cutoff
        h=dict(epoch=epoch,threshold=cutoff,positive_development_recall=recall)
        history.append(h)
        if epoch%10==0:print(json.dumps(h),flush=True)
    model.load_state_dict(state);model.threshold=threshold
    saved={**ckpt,'model_state':state,'architecture':'crossing_context_64','context_threshold':threshold,
        'preprocessing':{**ckpt['preprocessing'],'paired_context64':True},'training_kind':'frozen_base_context64','base_checkpoint':str(BASE)}
    torch.save(saved,run/'best_model.pt');dump(run/'history.json',history)
    dump(run/'config.json',dict(threshold=threshold,development_recall=best,parameters=sum(p.numel() for p in model.parameters()),
        original_cnn_frozen=True,limitations='Experimental full-frame context pass; FPGA export requires additional implementation. Same-map pose evaluation.'))
    write(run/'manifest.csv',selected);print('CONTEXT64_COMPLETE',threshold,best,flush=True)

if __name__=='__main__':main()
