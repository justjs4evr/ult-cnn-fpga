"""No motors/camera: controlled same-model Pi CPU and FPGA-receiver comparison."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import time
import numpy as np
import cv2

HERE=Path(__file__).resolve().parent
CONTEXT_PROBABILITY=0.9446295628547668  # Exact frozen checkpoint metadata.
CONTEXT_LOGIT=math.log(CONTEXT_PROBABILITY/(1-CONTEXT_PROBABILITY))

def system_state():
    state={}
    for name,cmd in [('throttled',['vcgencmd','get_throttled']),('temperature',['vcgencmd','measure_temp'])]:
        try:
            state[name]=subprocess.check_output(cmd,text=True,timeout=3).strip()
        except Exception as exc:state[name]=str(exc)
    frequencies={}
    for path in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_cur_freq'):
        try:frequencies[path.parts[-3]]=int(path.read_text())
        except OSError:pass
    state['cpu_frequency_khz']=frequencies
    try:state['load_average']=list(os.getloadavg())
    except OSError:pass
    return state

def percentile(values,p):
    return float(np.percentile(np.asarray(values),p))

def validate(net,data):
    outputs=[]
    for i in range(len(data['legacy'])):
        net.setInput(data['legacy'][i:i+1],'legacy')
        net.setInput(data['context'][i:i+1],'context')
        logits,ctx=net.forward(['raw_logits','context_logit'])
        if not np.allclose(logits,data['logits'][i:i+1],rtol=1e-4,atol=1e-3):
            raise RuntimeError('CPU logits do not match original torch fixture '+str(i))
        if not np.allclose(ctx,data['context_logit'][i:i+1],rtol=1e-4,atol=1e-3):
            raise RuntimeError('CPU context does not match original torch fixture '+str(i))
        outputs.append(dict(input=str(data['labels'][i]),raw_class=int(logits.argmax()),
                            context_logit=float(ctx.reshape(-1)[0]),
                            torch_logit_max_abs_error=float(np.max(np.abs(logits-data['logits'][i:i+1]))),
                            torch_context_abs_error=float(np.max(np.abs(ctx-data['context_logit'][i:i+1])))))
    return outputs

def run_cpu(model,data,threads,seconds,repeats):
    cv2.setNumThreads(threads)
    net=cv2.dnn.readNetFromONNX(str(model))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    validation=validate(net,data)
    for i in range(30):
        idx=i%len(data['legacy']);net.setInput(data['legacy'][idx:idx+1],'legacy');net.setInput(data['context'][idx:idx+1],'context')
        net.forward(['raw_logits','context_logit'])
    runs=[]
    for repeat in range(repeats):
        before=system_state();times=[];cpu_started=time.process_time();start=time.perf_counter();i=0
        while time.perf_counter()-start<seconds:
            idx=i%len(data['legacy'])
            t=time.perf_counter_ns()
            net.setInput(data['legacy'][idx:idx+1],'legacy')
            net.setInput(data['context'][idx:idx+1],'context')
            net.forward(['raw_logits','context_logit'])
            times.append((time.perf_counter_ns()-t)/1e6);i+=1
        wall=time.perf_counter()-start;cpu=time.process_time()-cpu_started
        runs.append(dict(repeat=repeat+1,iterations=i,wall_seconds=wall,process_cpu_seconds=cpu,
                         cpu_ms_per_frame=cpu*1000/i,one_core_cpu_percent=100*cpu/wall,
                         four_core_capacity_percent=100*cpu/wall/os.cpu_count(),
                         inference_ms=dict(mean=statistics.mean(times),median=statistics.median(times),
                                           p95=percentile(times,95),p99=percentile(times,99),maximum=max(times)),
                         state_before=before,state_after=system_state()))
        time.sleep(.2)
    return dict(threads=threads,validated_outputs=validation,runs=runs,
                median_run_mean_inference_ms=statistics.median(r['inference_ms']['mean'] for r in runs),
                median_cpu_ms_per_frame=statistics.median(r['cpu_ms_per_frame'] for r in runs))

def observe_receiver(seconds):
    # Only the result UART is opened read-only. This path never imports or opens UNO.
    root=Path(__file__).resolve().parents[1]/'rpi_receiver'
    import sys
    sys.path.insert(0,str(root))
    from receiver import Receiver,choose_port,open_readonly
    receiver=Receiver(json.loads((root/'manifest.json').read_text()))
    before=system_state()
    with open_readonly(choose_port(None),460800) as port:
        until=time.monotonic()+2
        while time.monotonic()<until:port.read(max(1,min(4096,port.in_waiting)))
        port.reset_input_buffer()
        start=time.perf_counter();cpu_start=time.process_time();events=0
        while time.perf_counter()-start<seconds:
            data=port.read(max(1,min(4096,port.in_waiting)))
            receiver.feed(data,time.monotonic());receiver.poll(time.monotonic())
        wall=time.perf_counter()-start;cpu=time.process_time()-cpu_start
    return dict(wall_seconds=wall,process_cpu_seconds=cpu,valid_frames=receiver.valid_frames,
                CRC_errors=receiver.decoder.crc_errors,rejected=receiver.rejected_frames,
                cpu_ms_per_result=cpu*1000/receiver.valid_frames if receiver.valid_frames else None,
                one_core_cpu_percent=100*cpu/wall,state_before=before,state_after=system_state(),
                motor_port_opened=False,scope='Pi receiver and decoder only; excludes UNO and logging IO')

def run_paced(model,data,seconds,interval=.31515005724468076):
    cv2.setNumThreads(1)
    net=cv2.dnn.readNetFromONNX(str(model))
    validate(net,data)
    before=system_state();count=0;latencies=[]
    start=time.perf_counter();cpu_start=time.process_time();next_frame=start
    while time.perf_counter()-start<seconds:
        now=time.perf_counter()
        if now<next_frame:
            time.sleep(min(next_frame-now,max(0,seconds-(now-start))))
            continue
        i=count%len(data['legacy']);t=time.perf_counter_ns()
        net.setInput(data['legacy'][i:i+1],'legacy');net.setInput(data['context'][i:i+1],'context')
        logits,ctx=net.forward(['raw_logits','context_logit'])
        raw_class=int(logits.argmax());activate=bool(ctx.reshape(-1)[0]>=CONTEXT_LOGIT)
        latencies.append((time.perf_counter_ns()-t)/1e6)
        count+=1;next_frame=start+count*interval
    wall=time.perf_counter()-start;cpu=time.process_time()-cpu_start
    return dict(frames=count,requested_interval_seconds=interval,wall_seconds=wall,
                process_cpu_seconds=cpu,cpu_ms_per_result=cpu*1000/count,
                one_core_cpu_percent=100*cpu/wall,
                inference_ms=dict(mean=statistics.mean(latencies),p95=percentile(latencies,95),maximum=max(latencies)),
                state_before=before,state_after=system_state(),
                scope='Single-thread DNN + argmax/context activation, paced to observed FPGA result interval; excludes capture and UNO')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds',type=float,default=3)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--receiver-seconds',type=float,default=10)
    ap.add_argument('--thread-counts',default='1,4')
    args=ap.parse_args()
    if not 1<=args.seconds<=10 or not 1<=args.repeats<=5 or not 1<=args.receiver_seconds<=30:
        ap.error('Bounded benchmark durations required')
    counts=[int(x) for x in args.thread_counts.split(',')]
    if not counts or any(n not in (1,4) for n in counts):ap.error('Use thread-counts1 or4')
    cv2.ocl.setUseOpenCL(False)
    model=HERE/'ult_cnn_dual.onnx';data=np.load(HERE/'fixtures.npz',allow_pickle=False)
    state=system_state()
    raw=int(state['throttled'].split('=')[1],16)
    if raw&15:raise RuntimeError('Current undervoltage/throttling detected before benchmark: '+state['throttled'])
    report=dict(status='running',platform=platform.platform(),python=platform.python_version(),
                opencv=cv2.__version__,numpy=np.__version__,cpu_cores=os.cpu_count(),
                opencl_enabled=cv2.ocl.useOpenCL(),
                model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                benchmark_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                context_activate_probability=CONTEXT_PROBABILITY,
                context_activate_logit=CONTEXT_LOGIT,
                electrical_condition='Tang remained powered from Pi USB and ran inference in both CPU and receiver conditions',
                state_before=state,motor_port_opened=False,camera_opened=False,
                scope='Same weights, dual preprocessed tensors. CPU=float32 OpenCV DNN; FPGA=signed16 fixed point.',
                cpu_tests=[],timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    try:
        for threads in counts:
            report['cpu_tests'].append(run_cpu(model,data,threads,args.seconds,args.repeats))
            print('Finished CPU thread count',threads,flush=True)
        report['paced_CPU']=run_paced(model,data,args.receiver_seconds)
        report['FPGA_receiver_only']=observe_receiver(args.receiver_seconds)
        report['state_after']=system_state()
        report['current_power_or_throttle_seen']=any(
            int(r[when]['throttled'].split('=')[1],16)&15
            for t in report['cpu_tests'] for r in t['runs'] for when in ('state_before','state_after'))!=0
        report['current_power_or_throttle_seen'] |= any(
            int(report[key][when]['throttled'].split('=')[1],16)&15
            for key in ('paced_CPU','FPGA_receiver_only') for when in ('state_before','state_after'))!=0
        report['status']='passed' if not report['current_power_or_throttle_seen'] else 'limited_by_power'
    except Exception as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        (HERE/'pi_cpu_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
