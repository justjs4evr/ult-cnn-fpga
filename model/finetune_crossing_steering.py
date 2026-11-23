"""Correct crossing steering using known simulator pose; keep six-class CNN.

The pose-derived labels are limited to straight crossing continuations.
No simulator coordinates or line tracker are used by the deployed controller.
Nearby poses on one map are not an independent generalization test.
"""
import argparse,copy,csv,hashlib,json,shutil
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from inference import CLASSES,load_model,preprocess_pil
from finetune_baseline_crossings import evaluate,key,dump,FLIP

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'user_runs/crossing_fix_20261011'
BASE=ROOT/'user_runs/crossing_bidirectional_20261010_215753_703782/best_model.pt'
ORIGINAL=ROOT/'user_runs/train_20261005_204140_992141'

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(path,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

def pixels(rows,pre):
    return np.stack([np.rint(preprocess_pil(Image.open(r['path']),pre)*255).astype('uint8') for r in rows])

def local_eval(model,x,y,rows,split):
    idx=[i for i,r in enumerate(rows) if r['split']==split]
    with torch.no_grad():pred=model(x[idx]).argmax(1).numpy()
    expected=y[idx].numpy();correct=(pred==expected)|((expected==5)&(pred==0))
    center=np.array([float(rows[i]['lateral_offset_mm'])==0 and float(rows[i]['yaw_offset_deg'])==0 for i in idx])
    return dict(count=len(idx),action_accuracy=float(correct.mean()),
        center_count=int(center.sum()),center_passage_correct=int(correct[center].sum()),
        predictions=dict(Counter(CLASSES[int(p)] for p in pred))),pred

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='training')
    ap.add_argument('--focus-weight',type=float,default=.85)
    ap.add_argument('--teacher-weight',type=float,default=.5)
    ap.add_argument('--road-weight',type=float,default=1.)
    ap.add_argument('--center-weight',type=float,default=0.)
    ap.add_argument('--region',choices=['all','north_entry'],default='all')
    ap.add_argument('--crop-top-ratio',type=float,default=None)
    ap.add_argument('--epochs',type=int,default=18)
    ap.add_argument('--dense-route',action='store_true')
    ap.add_argument('--road-ce-weight',type=float,default=0.)
    args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(111);np.random.seed(111)
    run=OUT/args.name;run.mkdir(exist_ok=False)
    teacher,ckpt=load_model(BASE);teacher_pre=ckpt['preprocessing'];pre=copy.deepcopy(teacher_pre)
    if args.crop_top_ratio is not None:pre['crop_top_ratio']=args.crop_top_ratio
    model=copy.deepcopy(teacher)
    teacher.requires_grad_(False)
    shutil.copy2(BASE,run/'baseline_model.pt');shutil.copy2(Path(__file__),run/Path(__file__).name)
    original=read(ORIGINAL/'split_manifest.csv');op=pixels(original,pre)
    teacher_op=(np.load(OUT/'preprocessing_cache.npz')['original'] if pre!=teacher_pre else op)
    oy=np.array([CLASSES.index(r['training_class']) for r in original])
    historical_held={key(p) for p,r in zip(op,original) if r['split']!='train'}
    new=read(OUT/'steering_capture/manifest.csv')
    if args.region=='north_entry':
        new=[r for r in new if r['direction']=='northbound' and float(r['map_y_px'])>=1990]
    npix=pixels(new,pre)
    # Conflicting identical 16x16 inputs cannot learn two different actions.
    actions=defaultdict(set)
    for p,r in zip(npix,new):
        mirrored=p[:,::-1].tobytes()<p.tobytes()
        label=CLASSES.index(r['training_class'])
        actions[key(p)].add(int(FLIP[label]) if mirrored else label)
    conflicts={k for k,v in actions.items() if len(v)>1}
    held={key(p) for p,r in zip(npix,new) if r['split']!='train'}
    excluded=[];selected=[];selected_pixels=[];seen=set()
    for r,p in zip(new,npix):
        k=key(p);reason=None
        if k in conflicts:reason='identical_input_conflicting_actions'
        elif r['split']=='train' and k in historical_held:reason='historical_holdout'
        elif r['split']=='train' and k in held:reason='local_holdout'
        elif r['split']=='train' and k in seen:reason='training_duplicate'
        if reason:excluded.append({**r,'reason':reason});continue
        if r['split']=='train':seen.add(k)
        selected.append({**r,'sha256':hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()});selected_pixels.append(p)
    write(run/'additional_manifest.csv',selected);write(run/'excluded_manifest.csv',excluded)
    write(run/'original_split_manifest.csv',original)
    xp=torch.from_numpy(np.stack(selected_pixels).astype('float32')[:,None]/255)
    yp=torch.tensor([CLASSES.index(r['training_class']) for r in selected])
    ni=[i for i,r in enumerate(selected) if r['split']=='train']
    nx=xp[ni];ny=yp[ni];nx=torch.cat([nx,nx.flip(3)]);ny=torch.cat([ny,FLIP[ny]])
    centers=[i for i in ni if float(selected[i]['lateral_offset_mm'])==0 and float(selected[i]['yaw_offset_deg'])==0]
    cx=xp[centers];cx=torch.cat([cx,cx.flip(3)])
    # Original historical train only; remove exact/mirror local heldout inputs.
    oi=[i for i,r in enumerate(original) if r['split']=='train' and key(op[i]) not in held]
    rx=torch.from_numpy(op[oi].astype('float32')[:,None]/255);ry=torch.tensor(oy[oi])
    rx=torch.cat([rx,rx.flip(3)]);ry=torch.cat([ry,FLIP[ry]])
    teacher_rx=torch.from_numpy(teacher_op[oi].astype('float32')[:,None]/255)
    teacher_rx=torch.cat([teacher_rx,teacher_rx.flip(3)])
    weights=len(ry)/(6*torch.bincount(ry,minlength=6).float())
    # Preserve actual successful curve camera outputs, not human labels.
    if args.dense_route:
        folder=OUT/'dense_teacher_route/teacher/original_start'
        road=[dict(path=str(p),split='dev_route' if i%10==0 else 'train') for i,p in enumerate(sorted(folder.glob('*.png')))]
    else:
        road=read(BASE.parent/'unlabelled_route_manifest.csv')
    road=[r for r in road if r['split']=='train']
    roadp=pixels(road,pre)
    allowed=[i for i,p in enumerate(roadp) if key(p) not in historical_held and key(p) not in held]
    road=[road[i] for i in allowed];roadp=roadp[allowed]
    ax=torch.from_numpy(roadp.astype('float32')[:,None]/255)
    write(run/'unlabelled_road_replay.csv',road)
    ax=torch.cat([ax,ax.flip(3)])
    teacher_road=pixels(road,teacher_pre)
    teacher_ax=torch.from_numpy(teacher_road.astype('float32')[:,None]/255);teacher_ax=torch.cat([teacher_ax,teacher_ax.flip(3)])
    with torch.no_grad():at=teacher(teacher_ax)
    config=dict(baseline=str(BASE),preprocessing=pre,classes=CLASSES,parameters=sum(p.numel() for p in model.parameters()),
        label_rule='Straight continuation: desired heading = forward heading + atan2(-lateral_offset_mm,200); minus yaw. <=4deg STOP/forward, <=20deg gentle, otherwise sharp.',
        region=args.region,data=dict(Counter(r['split'] for r in selected)),excluded=dict(Counter(r['reason'] for r in excluded)),
        loss=dict(original_ce=.65,pose_action_ce=args.focus_weight,original_kl=args.teacher_weight,
                  curve_kl=args.road_weight,curve_teacher_argmax_ce=args.road_ce_weight,center_stop_ce=args.center_weight),
        limitation='Same-map nearby-pose evaluation; source labels are simulation steering targets, not human adjudication.')
    dump(run/'run_config.json',config);print(json.dumps(config),flush=True)
    basedev,_,_=evaluate(teacher,teacher_op,oy,original,'dev_frame',256)
    teacher_lp=pixels(selected,teacher_pre);teacher_xp=torch.from_numpy(teacher_lp.astype('float32')[:,None]/255)
    baselocal,_=local_eval(teacher,teacher_xp,yp,selected,'dev_local')
    dump(run/'before_development.json',dict(historical=basedev,local=baselocal))
    opt=torch.optim.Adam(model.parameters(),lr=2e-4);history=[]
    best=-1;beststate=copy.deepcopy(model.state_dict());bestepoch=0
    for epoch in range(1,args.epochs+1):
        model.train();order=torch.randperm(len(ry));losses=[]
        for start in range(0,len(order),256):
            idx=order[start:start+256];bx=rx[idx];by=ry[idx]
            fi=torch.randint(len(nx),(128,));ai=torch.randint(len(ax),(96,))
            logits=model(bx)
            with torch.no_grad():tl=teacher(teacher_rx[idx])
            kl=F.kl_div(F.log_softmax(logits/2,1),F.softmax(tl/2,1),reduction='batchmean')*4
            road_logits=model(ax[ai])
            roadkl=F.kl_div(F.log_softmax(road_logits/2,1),F.softmax(at[ai]/2,1),reduction='batchmean')*4
            roadce=F.cross_entropy(road_logits,at[ai].argmax(1))
            center_logits=model(cx[torch.randint(len(cx),(64,))])
            center_loss=F.cross_entropy(center_logits,torch.full((64,),5))
            loss=.65*F.cross_entropy(logits,by,weight=weights)+args.focus_weight*F.cross_entropy(model(nx[fi]),ny[fi])+args.teacher_weight*kl+args.road_weight*roadkl+args.road_ce_weight*roadce+args.center_weight*center_loss
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
        model.eval();dev,_,_=evaluate(model,op,oy,original,'dev_frame',256)
        local,_=local_eval(model,xp,yp,selected,'dev_local')
        with torch.no_grad():agreement=float((model(ax).argmax(1)==at.argmax(1)).float().mean())
        ok=dev['macro_f1']>=basedev['macro_f1']-.015 and dev['accuracy']>=basedev['accuracy']-.015 and agreement>=.90
        rank=local['action_accuracy']
        if ok and rank>best:best=rank;beststate=copy.deepcopy(model.state_dict());bestepoch=epoch
        h=dict(epoch=epoch,loss=float(np.mean(losses)),historical_development=dev,local_development=local,curve_teacher_agreement=agreement,acceptable=ok)
        history.append(h);dump(run/'history.json',history)
        saved={**ckpt,'model_state':copy.deepcopy(model.state_dict()),'preprocessing':pre,'epoch':epoch,'training_kind':'crossing_pose_steering','run_config':config}
        torch.save(saved,run/f'epoch_{epoch:02d}.pt')
        print(f"epoch {epoch:02d} acc={dev['accuracy']:.4f} F1={dev['macro_f1']:.4f} local={rank:.4f} road={agreement:.3f} acceptable={ok}",flush=True)
    model.load_state_dict(beststate)
    saved={**ckpt,'model_state':beststate,'preprocessing':pre,'epoch':bestepoch,'training_kind':'crossing_pose_steering','run_config':config}
    torch.save(saved,run/'best_model.pt')
    oldtest,_,_=evaluate(teacher,teacher_op,oy,original,'test_img',256)
    newtest,_,_=evaluate(model,op,oy,original,'test_img',256)
    oldlocal,_=local_eval(teacher,teacher_xp,yp,selected,'test_local');newlocal,pred=local_eval(model,xp,yp,selected,'test_local')
    dump(run/'comparison.json',dict(best_epoch=bestepoch,previous=dict(test=oldtest,local_test=oldlocal),candidate=dict(test=newtest,local_test=newlocal)))
    write(run/'local_test_predictions.csv',[{**r,'predicted_class':CLASSES[int(p)]} for r,p in zip([r for r in selected if r['split']=='test_local'],pred)])
    print('TRAINING_COMPLETE',bestepoch,flush=True)

if __name__=='__main__':main()
