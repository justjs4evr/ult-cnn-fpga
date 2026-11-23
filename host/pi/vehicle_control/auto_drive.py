"""Bounded FPGA->Pi->UNO automatic control, with a separate lifted test mode."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sys
import time

from vehicle_control import HERE,MotorLink,Receiver,choose_port,open_readonly
from drive_protocol import validate_config,wheel_targets
from turn_decay import SharpTurnDecay


def validate_mode(mode,seconds,wheels_lifted,config):
    validate_config(config)
    if not math.isfinite(seconds) or seconds<=0:
        raise ValueError('Duration must be finite and positive')
    if mode=='lifted':
        if not wheels_lifted or config.get('lifted_wheel_mapping_verified') is not True:
            raise ValueError('Lifted test needs confirmed lifted wheels and verified channel mapping')
        if seconds>10 or config['cruise_pwm']>50:
            raise ValueError('Lifted automatic trial is limited to 10 seconds and PWM50')
    elif mode=='ground':
        if config.get('calibration_verified') is not True or config.get('ground_load_test_verified') is not True:
            raise ValueError('Ground operation needs actual ground-load calibration; lifted checks do not satisfy it')
        if seconds>600:
            raise ValueError('Ground session maximum is 600 seconds')
    else:
        raise ValueError('Unknown mode')


class Cadence:
    """Reject burst delivery; does not claim synchronized absolute image age."""
    def __init__(self):
        self.tick=None;self.at=None;self.first=None;self.streak=0
    def update(self,tick,now):
        if self.tick is None:
            good=True
        else:
            camera_elapsed=((tick-self.tick)&0xffffffff)/1000
            host_elapsed=now-self.at
            good=.1<=host_elapsed<=.75 and 0<camera_elapsed<1 and abs(camera_elapsed-host_elapsed)<=.25
        self.streak=self.streak+1 if good else 1
        if self.first is None or not good:self.first=now
        self.tick=tick;self.at=now
        return good,self.streak>=3 and now-self.first>=.4


class Session:
    def __init__(self,receiver,link,config,emit,stop_file):
        self.receiver=receiver;self.link=link;self.config=config;self.emit=emit;self.stop_file=stop_file
        self.cadence=Cadence();self.latest=None;self.armed=False;self.last_send=-float('inf')
        self.commands=0;self.frame_count=0;self.pivot_started_at=None
        self.turn_decay=SharpTurnDecay(config);self.last_turn_limited=False
    def step(self,data,now):
        if self.stop_file.exists():raise RuntimeError('User stop file received')
        if self.pivot_started_at is not None and now-self.pivot_started_at>=self.config['sharp_pivot_max_seconds']:
            raise RuntimeError('Mecanum pivot trial reached duration limit; stop and inspect heading')
        new_frame=False
        for event in self.receiver.feed(data,now):
            self.emit(dict(kind='FPGA_frame',frame_id=event.get('frame_id'),status=event['status'],
                raw_class=event.get('raw_class'),steering_class=event.get('steering_class'),
                slowdown_event=event.get('slowdown_event'),capture_tick_ms=event.get('capture_tick_ms')))
            if event['status']!='ready':raise RuntimeError('Rejected FPGA frame: '+str(event.get('reason')))
            self.turn_decay.observe(event,now)
            self.latest=event;self.frame_count+=1;new_frame=True
        if self.latest is None:return
        if self.receiver.policy.stale(now):raise TimeoutError('FPGA stream stale')
        if new_frame:
            cadence_ok,ready=self.cadence.update(self.latest['capture_tick_ms'],now)
            if self.armed and not cadence_ok:raise RuntimeError('Capture cadence/backlog mismatch')
            if not self.armed and ready:
                self.link.arm();self.armed=True;self.emit(dict(kind='armed',reason='three progressing live results'))
        if self.armed:
            target=wheel_targets(self.latest,self.config)
            target=self.turn_decay.apply(target,self.latest,now)
            limited=target.get('turn_limited',False)
            # Reduce yaw promptly at the limit even between normal100ms sends.
            if now-self.last_send<.1 and not (limited and not self.last_turn_limited):return
            self.link.drive(target['left'],target['right'])
            self.commands+=1;self.last_send=time.monotonic()
            self.last_turn_limited=limited
            if target['left']*target['right']<0:
                if self.pivot_started_at is None:self.pivot_started_at=now
            else:self.pivot_started_at=None
            self.emit(dict(kind='motor_ACK',left_PWM=target['left'],right_PWM=target['right'],
                reason=target['reason'],source_frame_id=self.latest['frame_id'],
                turn_limited=limited,requested_left_PWM=target.get('requested_left_PWM',target['left']),
                requested_right_PWM=target.get('requested_right_PWM',target['right'])))


def auto_motor_port(explicit):
    if explicit:return explicit
    from serial.tools import list_ports
    candidates=[p.device for p in list_ports.comports() if (p.vid,p.pid)==(0x1a86,0x7523)]
    if len(candidates)!=1:raise ValueError('Specify one verified vehicle port with --motor-port')
    return candidates[0]


def run(mode,seconds,wheels_lifted,motor_port,output=None):
    config=json.loads((HERE/'vehicle_config.json').read_text(encoding='utf-8'))
    validate_mode(mode,seconds,wheels_lifted,config)
    fpga_name=choose_port(None);motor_name=auto_motor_port(motor_port)
    if Path(fpga_name).resolve()==Path(motor_name).resolve():raise ValueError('Vehicle and FPGA ports must differ')
    folder=output or HERE/'runs'/datetime.now(timezone.utc).strftime('auto_%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True,exist_ok=True)
    stop_file=HERE/'AUTO_STOP'
    if stop_file.exists():raise ValueError('AUTO_STOP is present; clear it deliberately before starting another session')
    manifest=json.loads((HERE.parent/'rpi_receiver/manifest.json').read_text(encoding='utf-8'))
    receiver=Receiver(manifest)
    link=None;session=None;error=None;stop_ack=False;interrupted=False
    started=time.monotonic()
    with (folder/'control.jsonl').open('w',encoding='utf-8') as log:
        def emit(value):
            value['session_elapsed_seconds']=time.monotonic()-started
            log.write(json.dumps(value)+'\n');log.flush()
        try:
            link=MotorLink(motor_name)
            if link.exchange('C')!=['C',str(link.seq),'W4']:raise RuntimeError('Expected W4 vehicle firmware')
            with open_readonly(fpga_name,460800) as fpga:
                drain=time.monotonic()+2
                while time.monotonic()<drain:
                    if stop_file.exists():raise RuntimeError('User stop during drain')
                    fpga.read(max(1,min(4096,fpga.in_waiting)))
                fpga.reset_input_buffer()
                session=Session(receiver,link,config,emit,stop_file)
                trial_start=time.monotonic()
                while time.monotonic()-trial_start<seconds:
                    data=fpga.read(max(1,min(256,fpga.in_waiting)))
                    session.step(data,time.monotonic())
                    if not session.armed and time.monotonic()-trial_start>3:
                        raise TimeoutError('No stable FPGA stream; motor not started')
        except KeyboardInterrupt:
            interrupted=True
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            emit(dict(kind='aborted',reason=error))
        finally:
            if link is not None:
                try:link.close();stop_ack=True;emit(dict(kind='stopped',STOP_ACK=True))
                except Exception as exc:
                    error=(error+'; ' if error else '')+'stop ACK unavailable: '+str(exc)
    summary=dict(status='passed' if error is None and not interrupted and session and session.commands else 'stopped' if interrupted else 'failed',
        mode=mode,requested_observation_seconds=seconds,motor_port=motor_name,FPGA_port=fpga_name,
        valid_FPGA_frames=receiver.valid_frames,CRC_errors=receiver.decoder.crc_errors,
        rejected_frames=receiver.rejected_frames,motor_ACK_commands=session.commands if session else 0,
        STOP_ACK_verified=stop_ack,error=error,interrupted=interrupted,
        wheels_lifted_confirmed=wheels_lifted,ground_load_calibration_verified=config.get('ground_load_test_verified',False),
        camera_angle_validation_deferred=True,YOLO_deferred=True,power_issue_deferred=True,
        absolute_lane_image_age_verified=False,auto_start_on_boot_enabled=False,
        config=config,physical_wheel_observation=None)
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(folder=str(folder),**summary),ensure_ascii=False,indent=2))
    return 0 if summary['status']=='passed' else 1


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['lifted','ground','stop'])
    ap.add_argument('--seconds',type=float,default=5)
    ap.add_argument('--motor-port')
    ap.add_argument('--wheels-lifted',action='store_true')
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    if args.mode=='stop':
        (HERE/'AUTO_STOP').write_text('stop requested\n',encoding='ascii')
        print('Stop requested. The controller checks this file each read cycle.')
        return 0
    return run(args.mode,args.seconds,args.wheels_lifted,args.motor_port,args.output)


if __name__=='__main__':raise SystemExit(main())
