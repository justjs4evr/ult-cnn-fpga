"""Export the frozen model as explicit dual-input ONNX and paired test fixtures."""
from pathlib import Path
import hashlib
import json
import sys
REPO=Path(__file__).resolve().parents[3]  # repository root
sys.path.insert(0,str(REPO/'model'))
import numpy as np
import torch
import onnx
from onnx import helper, numpy_helper, TensorProto
from inference import load_model

OUT=Path(__file__).resolve().parent
OUT.mkdir(parents=True,exist_ok=True)
checkpoint=REPO/'model/user_runs/crossing_fix_20261011/best_model.pt'
model,meta=load_model(checkpoint)
state=model.state_dict()
keys={'conv1':'base.features.0','conv2':'base.features.3','fc1':'base.classifier.1',
      'fc2':'base.classifier.3','context':'context_head'}
initializers=[]
for name,key in keys.items():
    for suffix in ('weight','bias'):
        array=state[key+'.'+suffix].numpy().astype(np.float32)
        initializers.append(numpy_helper.from_array(array,name+'_'+suffix))
nodes=[]
def features(input_name,prefix):
    prior=input_name
    for i in (1,2):
        conv=f'{prefix}_conv{i}';relu=conv+'_relu';pool=conv+'_pool'
        nodes.append(helper.make_node('Conv',[prior,f'conv{i}_weight',f'conv{i}_bias'],[conv],
                                      kernel_shape=[3,3],strides=[1,1],pads=[1,1,1,1]))
        nodes.append(helper.make_node('Relu',[conv],[relu]))
        nodes.append(helper.make_node('MaxPool',[relu],[pool],kernel_shape=[2,2],strides=[2,2]))
        prior=pool
    flat=f'{prefix}_flat'
    nodes.append(helper.make_node('Flatten',[prior],[flat],axis=1))
    return flat
a=features('legacy','legacy')
c=features('context','context')
nodes.extend([
    helper.make_node('Gemm',[a,'fc1_weight','fc1_bias'],['fc1'],transB=1),
    helper.make_node('Relu',['fc1'],['fc1_relu']),
    helper.make_node('Gemm',['fc1_relu','fc2_weight','fc2_bias'],['raw_logits'],transB=1),
    helper.make_node('Gemm',[c,'context_weight','context_bias'],['context_logit'],transB=1)])
graph=helper.make_graph(nodes,'ULT_CNN_dual_input',[
    helper.make_tensor_value_info('legacy',TensorProto.FLOAT,[1,1,16,16]),
    helper.make_tensor_value_info('context',TensorProto.FLOAT,[1,1,64,64])],[
    helper.make_tensor_value_info('raw_logits',TensorProto.FLOAT,[1,6]),
    helper.make_tensor_value_info('context_logit',TensorProto.FLOAT,[1,1])],initializers)
net=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)],producer_name='AICAR verified comparison')
net.ir_version=8
onnx.checker.check_model(net)
onnx.save(net,OUT/'ult_cnn_dual.onnx')
legacy=[];contexts=[];labels=[];gold_logits=[];gold_context=[];integer_class=[];integer_context=[]
for folder in sorted((REPO/'verification/vectors').iterdir()):
    if not folder.is_dir():continue
    def read(name):
        return [int(x,16) for x in (folder/name).read_text().split()]
    a=np.array(read('legacy_input.hex'),dtype=np.float32).reshape(1,1,16,16)/255
    b=np.array(read('context_input.hex'),dtype=np.float32).reshape(1,1,64,64)/255
    with torch.inference_mode():
        logits=model.base(torch.from_numpy(a)).numpy()
        ctx=model.context_head(model.context_features(torch.from_numpy(b))).numpy()
    legacy.append(a);contexts.append(b);labels.append(folder.name)
    gold_logits.append(logits);gold_context.append(ctx)
    integer_class.append(read('raw_class.hex')[0])
    q=read('context_logit.hex')[0]
    integer_context.append(q-(1<<32) if q&(1<<31) else q)
np.savez(OUT/'fixtures.npz',legacy=np.concatenate(legacy),context=np.concatenate(contexts),
         logits=np.concatenate(gold_logits),context_logit=np.concatenate(gold_context),
         integer_class=np.array(integer_class),integer_context_q=np.array(integer_context),labels=np.array(labels))
manifest=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              onnx_sha256=hashlib.sha256((OUT/'ult_cnn_dual.onnx').read_bytes()).hexdigest(),
              fixtures_sha256=hashlib.sha256((OUT/'fixtures.npz').read_bytes()).hexdigest(),
              parameters=2977,fixtures=len(labels),inputs=['1x1x16x16','1x1x64x64'],
              outputs=['raw6logits','context_logit'],format='float32; shared convolution weights',
              source_vector_names=labels,opset=13,
              scope='Same model and preprocessed inputs; excludes capture, preprocessing and transport.')
(OUT/'export_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
