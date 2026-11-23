"""Two-camera observation and decision preview. Never opens the motor UART."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rpi_receiver'))
from receiver import Receiver, choose_port, open_readonly
from vision_worker import camera_worker


def power_health():
    try:
        raw = subprocess.check_output(['vcgencmd', 'get_throttled'], text=True, timeout=2).strip()
        value = int(raw.split('=')[1], 16)
        return dict(raw=raw, undervoltage_now=bool(value & 1),
                    throttled_now=bool(value & 4), current_flags_clear=(value & 15)==0)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return dict(error=str(exc), current_flags_clear=None)


def preview(lane, vision, now, vision_age_limit=3.):
    lane_good = lane is not None and lane['status'] == 'ready' and now-lane['received_monotonic'] < .75
    age = None
    if vision and vision.get('status') == 'ready':
        sensor_age = vision.get('sensor_age_at_capture_seconds')
        if sensor_age is not None and sensor_age >= 0:
            age = now - vision['captured_monotonic'] + sensor_age
    vision_good = age is not None and 0 <= age < vision_age_limit
    objects = vision.get('detections', []) if vision_good else []
    relevant = [d for d in objects if d['name'] in ('person', 'traffic light', 'stop sign')]
    # Mission action/ROI/distance/color rules have not been configured.
    return dict(lane_fresh=lane_good, object_result_fresh=vision_good,
        object_result_age_seconds=age, preview_steering=lane.get('steering_class') if lane_good else None,
        preview_speed_mode=('hold' if not (lane_good and vision_good) else
            'mission_review' if relevant else 'slow' if lane.get('slowdown_event') else 'cruise'),
        relevant_objects=relevant, raw_lane_class=lane.get('raw_class') if lane_good else None,
        motors_enabled=False, mission_policy_configured=False,
        traffic_light_color='not_implemented', barrier_model='not_trained',
        absolute_lane_image_age_verified=False)


def lane_worker(stop, shared, lock, folder):
    receiver = Receiver(json.loads((HERE.parent / 'rpi_receiver/manifest.json').read_text(encoding='utf-8')))
    stats = dict(status='starting', frames=0, crc_errors=0, rejected_frames=0, startup_frames=0)
    def update(event, now):
        event['received_monotonic'] = now
        with lock:
            shared['lane'] = event
    try:
        from protocol import Decoder
        startup = Decoder()
        with open_readonly(choose_port(None), 460800) as port, (folder/'lane.jsonl').open('w', encoding='utf-8') as log:
            deadline = time.monotonic()+2
            while time.monotonic() < deadline and not stop.is_set():
                stats['startup_frames'] += len(startup.feed(port.read(max(1, min(4096, port.in_waiting)))))
            stats['startup_crc_errors'] = startup.crc_errors
            port.reset_input_buffer()
            while not stop.is_set():
                data = port.read(max(1, min(4096, port.in_waiting)))
                now = time.monotonic()
                events = receiver.feed(data, now)
                stale = receiver.poll(now)
                if stale:
                    events.append(stale)
                for event in events:
                    update(event, now)
                    log.write(json.dumps(event)+'\n')
                if events:
                    log.flush()
        stats.update(status='completed', frames=receiver.valid_frames,
                     crc_errors=receiver.decoder.crc_errors, rejected_frames=receiver.rejected_frames)
    except Exception as exc:
        stats.update(status='error', error=f'{type(exc).__name__}: {exc}')
        update(dict(status='disconnected', steering_class=None), time.monotonic())
    finally:
        (folder/'lane_summary.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, default=30)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.seconds <= 0:
        ap.error('--seconds must be positive')
    folder = args.output or HERE/'runs'/datetime.now(timezone.utc).strftime('joint_%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True, exist_ok=True)
    try:
        choose_port(None)
    except ValueError as exc:
        report = dict(status='incomplete', phase='USB_preflight', error=str(exc),
                      camera_worker_started=False, motors_driven=False,
                      motor_serial_port_opened=False)
        (folder/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
        return 1
    ctx = mp.get_context('spawn')
    channel = ctx.Queue(maxsize=1)
    vision_stop = ctx.Event()
    lane_stop = threading.Event()
    shared, lock = {}, threading.Lock()
    thread = threading.Thread(target=lane_worker, args=(lane_stop, shared, lock, folder), daemon=True)
    worker = ctx.Process(target=camera_worker, args=(channel, vision_stop, str(folder/'vision')), daemon=True)
    start = time.monotonic()
    power_before = power_health()
    seen, fresh_both, vision, worker_problem = 0, 0, None, None
    thread.start()
    worker.start()
    try:
        with (folder/'combined.jsonl').open('w', encoding='utf-8') as log:
            while time.monotonic()-start < args.seconds:
                try:
                    while True:
                        value = channel.get_nowait()
                        if value['status'] == 'ready':
                            vision = value
                            seen += 1
                        else:
                            worker_problem = value
                            vision = None
                except queue.Empty:
                    pass
                if not worker.is_alive() and worker.exitcode is not None:
                    worker_problem = worker_problem or dict(status='worker_exited', exitcode=worker.exitcode)
                    vision = None
                with lock:
                    lane = shared.get('lane')
                now = time.monotonic()
                state = preview(lane, vision, now)
                state.update(elapsed_seconds=now-start, vision_worker_problem=worker_problem)
                fresh_both += bool(state['lane_fresh'] and state['object_result_fresh'])
                log.write(json.dumps(state)+'\n')
                log.flush()
                time.sleep(.1)
    except KeyboardInterrupt:
        pass
    finally:
        lane_stop.set()
        vision_stop.set()
        thread.join(timeout=3)
        worker.join(timeout=10)
        forced = worker.is_alive()
        if forced:
            worker.terminate()
            worker.join(timeout=3)
        channel.close()
    lane_summary = json.loads((folder/'lane_summary.json').read_text(encoding='utf-8')) if (folder/'lane_summary.json').exists() else {'status':'not_finished'}
    vision_rows = [json.loads(line) for line in (folder/'vision/vision.jsonl').read_text(encoding='utf-8').splitlines()] if (folder/'vision/vision.jsonl').exists() else []
    timings = [row['inference_and_postprocess_ms'] for row in vision_rows]
    summary = dict(status='passed' if lane_summary.get('status')=='completed' and lane_summary.get('frames', 0)>0 and
        lane_summary.get('crc_errors')==0 and lane_summary.get('rejected_frames')==0 and fresh_both>0 and not forced and not worker_problem else 'incomplete',
        observation_seconds=args.seconds, lane=lane_summary, vision_frames_completed=len(vision_rows),
        vision_results_observed=seen, simultaneous_fresh_observations=fresh_both,
        yolo_latency_ms=dict(min=min(timings), max=max(timings), mean=sum(timings)/len(timings)) if timings else None,
        worker_forced_termination=forced, worker_problem=worker_problem, motors_driven=False,
        motor_serial_port_opened=False, vehicle_mission_accuracy_verified=False,
        camera_B_angle_retest_deferred_by_user=True, yolo_model='official COCO YOLOv5n v7.0 ONNX640',
        opencv_threads=3, power_before=power_before, power_after=power_health(),
        source_A='OV5647 CSI camera on Raspberry Pi', source_B='OV3660 -> XIAO -> Tang CNN -> USB',
        policy='Observation/preview only; no mission rules or motor adapter configured')
    (folder/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0 if summary['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
