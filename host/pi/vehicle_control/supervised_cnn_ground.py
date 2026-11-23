"""Short supervised FPGA-CNN ground run; calibration gate remains unchanged."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from auto_drive import HERE, Session, Receiver, auto_motor_port
from vehicle_control import MotorLink, choose_port, open_readonly
from drive_protocol import validate_config


def run(seconds, clear_floor, motor_power_on, motor_port=None, output=None, config_path=None, fpga_port=None):
    if not clear_floor or not motor_power_on:
        raise ValueError('Require confirmed clear floor and motor power for supervised run')
    if not 0 < seconds <= 10:
        raise ValueError('Supervised CNN run is limited to 10 seconds')
    config_file = Path(config_path) if config_path else HERE/'vehicle_config.json'
    config = validate_config(json.loads(config_file.read_text(encoding='utf-8')))
    fpga_name = choose_port(fpga_port)
    motor_name = auto_motor_port(motor_port)
    folder = output or HERE/'runs'/datetime.now(timezone.utc).strftime('supervised_cnn_%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    stop_file = HERE/'AUTO_STOP'
    if stop_file.exists(): raise RuntimeError('AUTO_STOP is present')
    manifest = json.loads((HERE.parent/'rpi_receiver/manifest.json').read_text(encoding='utf-8'))
    receiver = Receiver(manifest)
    link = None; session = None; error = None; stop_ack = False
    started = time.monotonic()
    with (folder/'control.jsonl').open('w',encoding='utf-8') as log:
        def emit(value):
            value['session_elapsed_seconds'] = time.monotonic()-started
            log.write(json.dumps(value,ensure_ascii=False)+'\n'); log.flush()
        try:
            link = MotorLink(motor_name)
            if link.exchange('C') != ['C',str(link.seq),'W4']:
                raise RuntimeError('Expected W4 vehicle firmware')
            with open_readonly(fpga_name,460800) as fpga:
                drain=time.monotonic()+2
                while time.monotonic()<drain:
                    if stop_file.exists(): raise RuntimeError('User stop during drain')
                    fpga.read(max(1,min(4096,fpga.in_waiting)))
                fpga.reset_input_buffer()
                session=Session(receiver,link,config,emit,stop_file)
                trial_start=time.monotonic()
                while time.monotonic()-trial_start<seconds:
                    session.step(fpga.read(max(1,min(256,fpga.in_waiting))),time.monotonic())
                    if not session.armed and time.monotonic()-trial_start>3:
                        raise TimeoutError('No stable live FPGA cadence; motor not started')
        except KeyboardInterrupt:
            error='Interrupted by user'
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            emit(dict(kind='aborted',reason=error))
        finally:
            if link is not None:
                try: link.close(); stop_ack=True; emit(dict(kind='stopped',STOP_ACK=True))
                except Exception as exc: error=(error+'; ' if error else '')+'STOP ACK unavailable: '+str(exc)
    summary=dict(status='passed' if error is None and session and session.commands else 'failed',
        mode='supervised_cnn_ground',requested_observation_seconds=seconds,
        motor_port=motor_name,FPGA_port=fpga_name,valid_FPGA_frames=receiver.valid_frames,
        CRC_errors=receiver.decoder.crc_errors,rejected_frames=receiver.rejected_frames,
        motor_ACK_commands=session.commands if session else 0,STOP_ACK_verified=stop_ack,error=error,
        clear_floor_confirmed=clear_floor,motor_power_on_confirmed=motor_power_on,
        config_file=str(config_file),
        calibration_verified=config.get('calibration_verified',False),ground_load_test_verified=config.get('ground_load_test_verified',False),
        autonomous_driving=False,YOLO_deferred=True,camera_angle_validation_deferred=True,config=config)
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(folder=str(folder),**summary),ensure_ascii=False,indent=2))
    return 0 if summary['status']=='passed' else 1


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seconds',type=float,default=5)
    ap.add_argument('--clear-floor',action='store_true')
    ap.add_argument('--motor-power-on',action='store_true')
    ap.add_argument('--motor-port')
    ap.add_argument('--fpga-port')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--config',type=Path)
    args=ap.parse_args()
    return run(args.seconds,args.clear_floor,args.motor_power_on,args.motor_port,args.output,args.config,args.fpga_port)


if __name__=='__main__': raise SystemExit(main())
