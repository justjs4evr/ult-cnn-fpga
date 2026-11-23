"""Predict a new photo using the exported HEX files, not floating weights."""
import argparse
import json
from pathlib import Path
import numpy as np
from export_fpga import HERE,LAYERS,inputs,integer_forward


def read_signed(path,bits=16):
    values=np.array([int(line,16) for line in path.read_text(encoding='ascii').split()],dtype='int64')
    return np.where(values&(1<<(bits-1)),values-(1<<bits),values)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('image',type=Path)
    args=ap.parse_args();manifest=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    parameters={}
    for name,key in LAYERS:
        wt=next(t for t in manifest['tensors'] if t['name']==key+'.weight')
        bt=next(t for t in manifest['tensors'] if t['name']==key+'.bias')
        parameters[name]=(read_signed(HERE/wt['file']).reshape(wt['shape']),
            read_signed(HERE/bt['file']).reshape(bt['shape']),wt['fractional_bits'])
    x,c,_=inputs(args.image,manifest['preprocessing']);stats={}
    a,b,g=integer_forward(x[None],c[None],parameters,stats)
    raw=int(a['logits'][0].argmax());q=int(g[0]);activate=q>=manifest['thresholds']['activate_q']
    print(json.dumps(dict(raw_class=manifest['classes'][raw],class_result='stop' if activate else manifest['classes'][raw],
        context_logit_q=q,context_activate=activate,context_keep=q>=manifest['thresholds']['keep_q'],
        note='Single photo; temporal steering policy is in rpi_receiver.py'),ensure_ascii=False))


if __name__=='__main__':main()
