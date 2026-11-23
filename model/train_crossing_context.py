"""Experimental context head; select threshold on dev, never replace user model."""
import csv,json,copy,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from inference import CLASSES,load_model
from finetune_crossing_steering import OUT,BASE,ORIGINAL,read,pixels,write
from finetune_baseline_crossings import key,dump
from crossing_context_model import CrossingContextCNN

def main():
    torch.set_num_threads(4);torch.manual_seed(121)
    run=OUT/'context_candidate';run.mkdir(exist_ok=False)
    old,ckpt=load_model(BASE);pre=ckpt['preprocessing'];full={**pre,'crop_top_ratio':0.}
    model=CrossingContextCNN();model.base.load_state_dict(old.state_dict());model.base.requires_grad_(False)
    orig=read(ORIGINAL/'split_manifest.csv');crop=np.load(OUT/'preprocessing_cache.npz')['original'];raw=pixels(orig,full)
    new=read(OUT/'steering_capture/manifest.csv');nc=pixels(new,pre);nf=pixels(new,full)
    historical_held={key(p) for p,r in zip(crop,orig) if r['split']!='train'}
    local_held={key(p) for p,r in zip(nc,new) if r['split']!='train'}
    rows=[];features=[];labels=[]
    # Cropped features preserve alignment cues, full features retain crossing context.
    def featurize(c,f):
        x=torch.from_numpy(np.stack([c,f],1).astype('float32')/255)
        with torch.no_grad():return torch.cat([model.context_features(x[i:i+512]) for i in range(0,len(x),512)])
    of=featurize(crop,raw);newf=featurize(nc,nf)
    for i,r in enumerate(orig):
        if r['split']=='train' and key(crop[i]) in local_held:continue
        rows.append(dict(path=r['path'],source='historical',split=r['split']))
        features.append(of[i]);labels.append(int(r['source_class'] in ('crossroads','crosswalk')))
    for i,r in enumerate(new):
        if r['split']=='train' and (key(nc[i]) in historical_held or key(nc[i]) in local_held):continue
        rows.append(dict(path=r['path'],source='simulated',split=r['split']))
        features.append(newf[i]);labels.append(int(abs(float(r['yaw_offset_deg']))<=5))
    features=torch.stack(features);labels=torch.tensor(labels,dtype=torch.float32)
    train=torch.tensor([i for i,r in enumerate(rows) if r['split']=='train'])
    dev=torch.tensor([i for i,r in enumerate(rows) if r['split'] in ('dev_frame','dev_local')])
    positive=train[labels[train]==1];negative=train[labels[train]==0]
    # Explicit normal-route negatives: successful road frames before the crossing.
    roads=read(BASE.parent/'unlabelled_route_manifest.csv');rc=pixels(roads,pre);rf=pixels(roads,full);roadf=featurize(rc,rf)
    rtrain=torch.tensor([i for i,r in enumerate(roads) if r['split']=='train']);rdev=torch.tensor([i for i,r in enumerate(roads) if r['split']!='train'])
    optimizer=torch.optim.Adam(model.context_head.parameters(),lr=.002);history=[];best=-1;state=None;threshold=None
    for epoch in range(1,61):
        for _ in range(100):
            pi=positive[torch.randint(len(positive),(64,))];ni=negative[torch.randint(len(negative),(192,))]
            ri=rtrain[torch.randint(len(rtrain),(64,))]
            x=torch.cat([features[pi],features[ni],roadf[ri]])
            y=torch.cat([torch.ones(64),torch.zeros(256)])
            loss=F.binary_cross_entropy_with_logits(model.context_head(x).flatten(),y)
            optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():dp=model.context_head(features[dev]).flatten().sigmoid();rp=model.context_head(roadf[rdev]).flatten().sigmoid()
        # No additional STOP predictions on the historical development negatives
        # or the normal-curve replay development set.
        cutoff=max(.80,float(dp[labels[dev]==0].max())+.002,float(rp.max())+.002)
        positives_dev=labels[dev]==1
        recall=float((dp[positives_dev]>=cutoff).float().mean()) if cutoff<1 else 0.
        if recall>best:best=recall;state=copy.deepcopy(model.state_dict());threshold=cutoff
        h=dict(epoch=epoch,threshold=cutoff,positive_development_recall=recall,road_development_max_probability=float(rp.max()))
        history.append(h)
        if epoch%10==0:print(json.dumps(h),flush=True)
    model.load_state_dict(state);model.threshold=threshold
    paired={**pre,'paired_full_frame_context':True}
    saved={**ckpt,'model_state':state,'preprocessing':paired,'architecture':'crossing_context_shared','context_threshold':threshold,
           'training_kind':'frozen_base_crossing_context','base_checkpoint':str(BASE)}
    torch.save(saved,run/'best_model.pt');dump(run/'history.json',history)
    dump(run/'config.json',dict(threshold=threshold,development_recall=best,parameters=sum(p.numel() for p in model.parameters()),
        original_cnn_frozen=True,limitations='Experimental paired-input CNN; additional FPGA preprocessing/export work needed. Same-map testing only.'))
    write(run/'manifest.csv',[{**r,'context_label':int(labels[i])} for i,r in enumerate(rows)])
    print('CONTEXT_COMPLETE',threshold,best,flush=True)

if __name__=='__main__':main()
