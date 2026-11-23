"""Export the exact crossing_context_64 checkpoint and validate fixed-point outputs.

Execute with the existing ULT_CNN .venv. This package is an arithmetic/RTL
verification release, not a camera-integrated board bitstream.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from inference import load_model, image_tensor, CLASSES

CHECKPOINT = ROOT / 'user_runs/crossing_fix_20261011/best_model.pt'
RUN = CHECKPOINT.parent
LAYERS = [('conv1', 'base.features.0'), ('conv2', 'base.features.3'),
          ('fc1', 'base.classifier.1'), ('fc2', 'base.classifier.3'), ('context', 'context_head')]
ACT_BITS = 11
SCALE = 1 << ACT_BITS
OUTPUT_BITS = dict(conv1=11, conv2=10, fc1=8, fc2=9, context=11)
INPUT_BITS = dict(conv1=11, conv2=11, fc1=10, fc2=8, context=10)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def round_away(x):
    return np.sign(x) * np.floor(np.abs(x) + .5)


def shift_away(x, bits):
    return np.where(x < 0, -((np.abs(x) + (1 << (bits-1))) >> bits),
                    (x + (1 << (bits-1))) >> bits)


def write_hex(path, values, digits=4):
    mask = (1 << (digits*4))-1
    path.write_text(''.join(f'{int(x)&mask:0{digits}X}\n' for x in np.asarray(values).reshape(-1)), encoding='ascii')


def params(metadata):
    result = {}
    tensors = []
    for name, key in LAYERS:
        wf = metadata['model_state'][key+'.weight'].numpy().astype('float64')
        bf = metadata['model_state'][key+'.bias'].numpy().astype('float64')
        bits = 14
        while np.abs(round_away(wf*(1 << bits))).max() > 32767:
            bits -= 1
        w = round_away(wf*(1 << bits)).astype('int64')
        b = round_away(bf*(1 << OUTPUT_BITS[name])).astype('int64')
        if np.abs(b).max() > 32767:
            raise ValueError('Bias overflow')
        result[name] = (w, b, bits)
        for suffix, value in [('w', w), ('b', b)]:
            filename = f'{name}_{suffix}.hex'
            write_hex(HERE/'weights'/filename, value)
            tensors.append(dict(name=key+('.weight' if suffix=='w' else '.bias'),
                file='weights/'+filename, shape=list(value.shape), count=value.size,
                fractional_bits=bits if suffix=='w' else OUTPUT_BITS[name],
                min=int(value.min()), max=int(value.max())))
    return result, tensors


def requant(mac, bias, bits, relu, stats, key, wide=False):
    stats[key+'_mac_abs_max'] = max(stats.get(key+'_mac_abs_max', 0), int(np.abs(mac).max()))
    y = shift_away(mac, bits) + bias
    if relu:
        y = np.maximum(y, 0)
    stats[key+'_abs_max_before_clip'] = max(stats.get(key+'_abs_max_before_clip', 0), int(np.abs(y).max()))
    lo, hi = (-2147483648, 2147483647) if wide else (-32768, 32767)
    stats[key+'_saturated'] = stats.get(key+'_saturated', 0) + int(((y < lo) | (y > hi)).sum())
    return np.clip(y, lo, hi)


def conv(x, p, stats, key):
    w, b, bits = p
    padded = np.pad(x, ((0,0), (0,0), (1,1), (1,1)))
    patches = sliding_window_view(padded, (3,3), axis=(2,3))
    n, _, h, width, _, _ = patches.shape
    matrix = patches.transpose(0,2,3,1,4,5).reshape(-1, w.shape[1]*9)
    mac = (matrix @ w.reshape(w.shape[0], -1).T).reshape(n,h,width,w.shape[0]).transpose(0,3,1,2)
    name='conv1' if key.endswith('conv1') else 'conv2'
    shift=bits+INPUT_BITS[name]-OUTPUT_BITS[name]
    return requant(mac, b[None,:,None,None], shift, True, stats, key)


def pool(x):
    n,c,h,w=x.shape
    return x.reshape(n,c,h//2,2,w//2,2).max(axis=(3,5))


def features(x, parameters, stats, prefix):
    c1=conv(x,parameters['conv1'],stats,prefix+'conv1');p1=pool(c1)
    c2=conv(p1,parameters['conv2'],stats,prefix+'conv2');p2=pool(c2)
    return dict(conv1=c1,pool1=p1,conv2=c2,pool2=p2)


def dense(x,p,relu,stats,key,wide=False):
    w,b,bits=p
    name='context' if key=='context_head' else key
    return requant(x@w.T,b[None],bits+INPUT_BITS[name]-OUTPUT_BITS[name],relu,stats,key,wide)


def integer_forward(legacy_bytes, context_bytes, parameters, stats):
    legacy=((legacy_bytes.astype('int64')*SCALE+127)//255)[:,None]
    context=((context_bytes.astype('int64')*SCALE+127)//255)[:,None]
    a=features(legacy,parameters,stats,'legacy_')
    a['fc1']=dense(a['pool2'].reshape(-1,96),parameters['fc1'],True,stats,'fc1')
    a['logits']=dense(a['fc1'],parameters['fc2'],False,stats,'fc2')
    b=features(context,parameters,stats,'context_')
    logit=dense(b['pool2'].reshape(-1,1536),parameters['context'],False,stats,'context_head',True).reshape(-1)
    return a,b,logit


def inputs(path, preprocessing):
    with Image.open(path) as im:
        tensor=image_tensor(im,preprocessing)
    legacy=np.rint(tensor[0,0,::4,::4].numpy()*255).astype('uint8')
    context=np.rint(tensor[0,1].numpy()*255).astype('uint8')
    return legacy,context,tensor


def rows_from_manifest(path, split):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return [r for r in csv.DictReader(stream) if r['split']==split]


def validation_rows():
    rows=[dict(path=r['path'],group='historical_test',truth=r['training_class'])
          for r in rows_from_manifest(RUN/'split_manifest.csv','test_img')]
    for group in ('augmented_context_test','fresh_context_test'):
        for path in sorted((RUN/group/'candidate').glob('*/*.png')):
            rows.append(dict(path=str(path),group=group,truth=''))
    # Successful route camera PNGs have their own group; no teacher labels are used.
    for path in sorted((RUN/'dense_teacher_route/teacher/original_start').glob('*.png')):
        rows.append(dict(path=str(path),group='normal_route_replay',truth=''))
    native=ROOT/'webots_runs/ult_context_runtime_validation/20261011_065750_509210'
    for path in sorted(native.glob('frame_*.jpg')):
        rows.append(dict(path=str(path),group='actual_controller_jpeg_replay',truth=''))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch-size',type=int,default=32);args=ap.parse_args()
    torch.set_num_threads(4)
    model,metadata=load_model(CHECKPOINT)
    if metadata.get('architecture')!='crossing_context_64':raise ValueError('Unsupported architecture')
    (HERE/'weights').mkdir(exist_ok=True)
    parameters,tensors=params(metadata)
    activate=math.log(metadata['context_policy']['activate_probability']/(1-metadata['context_policy']['activate_probability']))
    keep=math.log(.9/.1)
    # Compare integer logit directly; sigmoid is monotonic, so no RTL exp/divide needed.
    thresholds=dict(activate_q=math.ceil(activate*SCALE),keep_q=math.ceil(keep*SCALE),
                    activate_probability=metadata['context_policy']['activate_probability'],keep_probability=.9,
                    fractional_bits=ACT_BITS,clear_seconds=1.5)
    config=dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        architecture=metadata['architecture'],classes=list(CLASSES),parameter_count=2977,
        input_fractional_bits=ACT_BITS,activation_fractional_bits=OUTPUT_BITS,weight_bits=16,activation_bits=16,context_logit_bits=32,
        accumulator_bits=64,rounding='nearest, ties away from zero for parameters and MAC shifts',
        input_conversion=f'(byte*{SCALE}+127)//255',tensor_order='Conv O,I,Y,X; dense O,I; features C,Y,X',
        saturation='int16 activations/logits; int32 context head logit',tensors=tensors,
        thresholds=thresholds,preprocessing=metadata['preprocessing'],
        status='fixed_point_and_RTL_verification_only_not_board_bitstream',
        preprocess_contract='Legacy: PIL L, crop top floor(H*.25), BOX64, >=127 binary, BOX16. Context: uncropped PIL L, BOX64. Both uint8.',
        rpi_contract='Receive raw legacy class and context logit; use two logit thresholds and 1.5 second grace. STOP currently maps to straight; motor mapping stays on RPi.')
    dump(HERE/'manifest.json',config)
    defines=[f'`define ACT_FRAC {ACT_BITS}',f'`define CONTEXT_ACTIVATE_Q {thresholds["activate_q"]}',f'`define CONTEXT_KEEP_Q {thresholds["keep_q"]}']
    defines += [f'`define {name.upper()}_W_FRAC {parameters[name][2]}' for name,_ in LAYERS]
    defines += [f'`define {name.upper()}_SHIFT {parameters[name][2]+INPUT_BITS[name]-OUTPUT_BITS[name]}' for name,_ in LAYERS]
    (HERE/'rtl/quant_config.vh').write_text('\n'.join(defines)+'\n',encoding='ascii')
    rows=validation_rows();predictions=[];stats={};vectors=[];saved=set()
    for start in range(0,len(rows),args.batch_size):
        chunk=rows[start:start+args.batch_size]
        prepared=[inputs(r['path'],metadata['preprocessing']) for r in chunk]
        x=np.stack([v[0] for v in prepared]);c=np.stack([v[1] for v in prepared])
        a,b,gate=integer_forward(x,c,parameters,stats)
        with torch.inference_mode():
            tx=torch.cat([v[2] for v in prepared]);legacy_float=model.base(tx[:,0:1,::4,::4]).numpy()
            gate_float=model.context_head(model.context_features(tx[:,1:2])).reshape(-1).numpy()
        raw=a['logits'].argmax(1);fr=legacy_float.argmax(1)
        for j,row in enumerate(chunk):
            fp=float(gate_float[j]);iq=int(gate[j]);fa=fp>=activate;ia=iq>=thresholds['activate_q']
            fk=fp>=keep;ik=iq>=thresholds['keep_q']
            final_int=5 if ia else int(raw[j]);final_float=5 if fa else int(fr[j])
            record=dict(**row,float_raw_class=CLASSES[fr[j]],integer_raw_class=CLASSES[raw[j]],
                float_final_class=CLASSES[final_float],integer_final_class=CLASSES[final_int],
                float_context_logit=fp,integer_context_logit_q=iq,
                context_logit_error=iq/SCALE-fp,activate_match=fa==ia,keep_match=fk==ik,
                raw_match=int(raw[j])==int(fr[j]),final_match=final_int==final_float)
            predictions.append(record)
            tag=None
            if row['group']=='historical_test' and int(raw[j])==int(fr[j]) and row['truth']==CLASSES[raw[j]]:
                tag='class_'+row['truth']
            elif row['group']=='augmented_context_test' and ia:tag='crossing_activate'
            elif row['group']=='normal_route_replay' and not ik:tag='context_clear'
            if tag and tag not in saved:
                saved.add(tag);folder=HERE/'vectors'/tag;folder.mkdir(parents=True,exist_ok=True)
                write_hex(folder/'legacy_input.hex',x[j],2);write_hex(folder/'context_input.hex',c[j],2)
                for k,v in a.items():write_hex(folder/('legacy_'+k+'.hex'),v[j])
                for k,v in b.items():write_hex(folder/('context_'+k+'.hex'),v[j])
                write_hex(folder/'context_logit.hex',[iq],8);write_hex(folder/'raw_class.hex',[raw[j]],2)
                write_hex(folder/'class_result.hex',[final_int],2)
                vectors.append(dict(name=tag,path=str(folder.relative_to(HERE)),source=row['path']))
        if start%256==0:print(f'Validated {min(start+len(chunk),len(rows))}/{len(rows)}',flush=True)
    with (HERE/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    summary={}
    for group in dict.fromkeys(p['group'] for p in predictions):
        subset=[p for p in predictions if p['group']==group]
        g=dict(images=len(subset),raw_class_matches=sum(p['raw_match'] for p in subset),
            final_class_matches=sum(p['final_match'] for p in subset),
            activate_matches=sum(p['activate_match'] for p in subset),keep_matches=sum(p['keep_match'] for p in subset),
            max_context_logit_error=max(abs(p['context_logit_error']) for p in subset))
        if group=='historical_test':
            g['float_accuracy']=sum(p['truth']==p['float_final_class'] for p in subset)/len(subset)
            g['integer_accuracy']=sum(p['truth']==p['integer_final_class'] for p in subset)/len(subset)
        summary[group]=g
    worst_macs=max(v for k,v in stats.items() if k.endswith('_mac_abs_max'))
    if worst_macs >= 2**63:raise OverflowError('64-bit accumulator overflow')
    saturation={k:v for k,v in stats.items() if k.endswith('_saturated')}
    report=dict(groups=summary,statistics=stats,saturation=saturation,vectors=vectors,
        input_count=len(predictions),status='numerical_comparison_complete',
        limitations=['All saved images are replayed, not new closed-loop integer-model driving.',
            'JPEG runtime snapshots differ from native camera bytes and are labelled replay only.',
            'No camera preprocessing RTL, CDC, UART physical interface, pin assignment or board bitstream in this release.'])
    dump(HERE/'quantization_report.json',report)
    dump(HERE/'vectors/manifest.json',vectors)
    print(json.dumps(dict(groups=summary,saturation=saturation,vectors=len(vectors)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
