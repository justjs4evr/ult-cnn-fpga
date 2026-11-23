"""Supplement real-image reference vectors with deterministic edge cases."""
import json
import numpy as np
from export_fpga import HERE,CHECKPOINT,load_model,params,integer_forward,write_hex,dump


def main():
    _,metadata=load_model(CHECKPOINT);parameters,_=params(metadata)
    m=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    cases={}
    cases['stress_zero']=(np.zeros((16,16),dtype='uint8'),np.zeros((64,64),dtype='uint8'))
    cases['stress_white']=(np.full((16,16),255,dtype='uint8'),np.full((64,64),255,dtype='uint8'))
    cases['stress_checker']=tuple(((np.indices((s,s)).sum(0)%2)*255).astype('uint8') for s in (16,64))
    rng=np.random.default_rng(20261025)
    cases['stress_random']=tuple(rng.integers(0,256,(s,s),dtype='uint8') for s in (16,64))
    edge=[]
    for s in (16,64):
        x=np.zeros((s,s),dtype='uint8');x[0,:]=255;x[-1,:]=255;x[:,0]=255;x[:,-1]=255;edge.append(x)
    cases['stress_border']=edge
    vectors=json.loads((HERE/'vectors/manifest.json').read_text(encoding='utf-8'))
    vectors=[r for r in vectors if not r['name'].startswith('stress_')]
    stats={}
    for name,(legacy,context) in cases.items():
        a,b,g=integer_forward(legacy[None],context[None],parameters,stats)
        path=HERE/'vectors'/name;path.mkdir(exist_ok=True)
        write_hex(path/'legacy_input.hex',legacy,2);write_hex(path/'context_input.hex',context,2)
        for k,v in a.items():write_hex(path/('legacy_'+k+'.hex'),v[0])
        for k,v in b.items():write_hex(path/('context_'+k+'.hex'),v[0])
        raw=int(a['logits'][0].argmax());final=5 if int(g[0])>=m['thresholds']['activate_q'] else raw
        write_hex(path/'context_logit.hex',g,8);write_hex(path/'raw_class.hex',[raw],2)
        write_hex(path/'class_result.hex',[final],2)
        vectors.append(dict(name=name,path='vectors/'+name,source='synthetic diagnostic, not a labelled driving image'))
    dump(HERE/'vectors/manifest.json',vectors)
    dump(HERE/'stress_vector_statistics.json',stats)
    print('Added 5 stress vectors; total',len(vectors))


if __name__=='__main__':main()
