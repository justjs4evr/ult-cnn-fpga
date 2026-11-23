"""Evaluate a paired-context candidate on the fixed original photo holdout."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from inference import CLASSES,load_model
from finetune_crossing_steering import OUT,ORIGINAL,BASE,read,write
from train_crossing_context64 import gray64
from train_classified_images import score

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(4);old,_=load_model(BASE);new,_=load_model(args.candidate)
    orig=read(ORIGINAL/'split_manifest.csv');allcrop=np.load(OUT/'preprocessing_cache.npz')['original']
    idx=np.array([i for i,r in enumerate(orig) if r['split']=='test_img']);rows=[orig[i] for i in idx]
    crop=allcrop[idx].astype('float32')/255;full=gray64(rows)
    x=torch.from_numpy(np.stack([np.repeat(np.repeat(crop,4,1),4,2),full],1))
    y=torch.tensor([CLASSES.index(r['training_class']) for r in rows])
    a,pa=score(old,torch.from_numpy(crop[:,None]),y,256,CLASSES);b,pb=score(new,x,y,128,CLASSES)
    preserved=all(torch.equal(v,new.base.state_dict()[k]) for k,v in old.state_dict().items())
    result=dict(previous=a,candidate=b,original_weights_exactly_preserved=preserved,
        limitations='Fixed original photo test; same-map simulation trials are separate and do not prove new-map or real-camera generalization.')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'photo_test_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    changed=[{**r,'previous_class':CLASSES[int(p)],'candidate_class':CLASSES[int(q)],
              'same_motor_action_if_STOP_bypassed':int(p) in (0,5) and int(q) in (0,5)}
             for r,p,q in zip(rows,pa,pb) if p!=q]
    write(args.output/'changed_photo_predictions.csv',changed)
    print('Photo accuracy',a['accuracy'],b['accuracy'],'macro F1',a['macro_f1'],b['macro_f1'],'base weights equal',preserved,flush=True)

if __name__=='__main__':main()
