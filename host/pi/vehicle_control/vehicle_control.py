"""FPGA-to-wheel preview plus explicit, bounded lifted-wheel bench support.

Default preview never opens or writes the vehicle port. YOLO is deferred.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import secrets
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'rpi_receiver'))
from receiver import Receiver,choose_port,open_readonly
from drive_protocol import Decoder,encode,wheel_targets,validate_config


class MotorLink:
    def __init__(self,port):
        import serial
        self.port=serial.Serial(port=None,baudrate=115200,timeout=.02,write_timeout=.1)
        self.port.dtr=False;self.port.rts=False;self.port.port=port
        self.decoder=Decoder();self.seq=0;self.token=secrets.randbelow(0xffffffff)+1;self.pwm_limit=None
        try:
            self.port.open()
            # Opening can reset a UNO even with control lines preconfigured.
            time.sleep(2)
            self.port.reset_input_buffer()
            response=self.exchange('Q')
            if (len(response)!=6 or response[:3]!=['I',str(self.seq),'ULTDRV1'] or
                response[4:]!=['400','1'] or response[3] not in ('75','120')):
                raise RuntimeError('Expected new firmware ID and PCA9685 ready; no arm sent: '+repr(response))
            self.pwm_limit=int(response[3])
        except BaseException:
            self.port.close()
            raise
    def exchange(self,command,*fields):
        self.seq=(self.seq+1)&65535
        packet=encode(command,self.seq,*fields)
        self.port.write(packet);self.port.flush()
        deadline=time.monotonic()+.15
        while time.monotonic()<deadline:
            for response in self.decoder.feed(self.port.read(max(1,min(128,self.port.in_waiting)))):
                if len(response)>1 and response[1]==str(self.seq):
                    if response[0]=='E':
                        raise RuntimeError('Vehicle rejected command: '+repr(response))
                    return response
        raise TimeoutError('No matching vehicle acknowledgement')
    def arm(self):
        if self.exchange('A',self.token)!=['K',str(self.seq),'1','0','0']:
            raise RuntimeError('Invalid arm acknowledgement')
    def drive(self,left,right):
        limit=self.pwm_limit or 75
        if type(left) is not int or type(right) is not int or max(abs(left),abs(right))>limit:
            raise ValueError('PWM outside firmware limits')
        response=self.exchange('D',self.token,left,right)
        if response!=['K',str(self.seq),'1',str(left),str(right)]:
            raise RuntimeError('Wheel acknowledgement mismatch')
    def drive_wheels(self,values):
        limit=self.pwm_limit or 75
        if len(values)!=4 or any(type(v) is not int or abs(v)>limit for v in values):
            raise ValueError('Require four integer wheel targets in -120..120')
        response=self.exchange('W',self.token,*values)
        if response!=['V',str(self.seq),'1',*[str(v) for v in values]]:
            raise RuntimeError('Four-wheel acknowledgement mismatch')
    def close(self):
        try:
            response=self.exchange('S')
            if response!=['K',str(self.seq),'0','0','0']:
                raise RuntimeError('Stop acknowledgement mismatch')
        finally:self.port.close()


def preview(seconds,folder,config,fpga_port=None):
    folder.mkdir(parents=True,exist_ok=True)
    receiver=Receiver(json.loads((HERE.parent/'rpi_receiver/manifest.json').read_text(encoding='utf-8')))
    count=0
    with open_readonly(choose_port(fpga_port),460800) as port,(folder/'preview.jsonl').open('w',encoding='utf-8') as log:
        # Drain historical USB data without creating motor requests.
        until=time.monotonic()+2
        while time.monotonic()<until:port.read(max(1,min(4096,port.in_waiting)))
        port.reset_input_buffer()
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            data=port.read(max(1,min(4096,port.in_waiting)))
            now=time.monotonic()
            events=receiver.feed(data,now)
            stale=receiver.poll(now)
            if stale:events.append(stale)
            for event in events:
                target=wheel_targets(event,config)
                row=dict(event=event,proposed_wheel_PWM=target,motor_port_opened=False,
                         motor_serial_bytes_sent=0,calibration_verified=config['calibration_verified'])
                log.write(json.dumps(row)+'\n');log.flush()
                count+=event['status']=='ready'
    result=dict(valid_frames=count,CRC_errors=receiver.decoder.crc_errors,rejected=receiver.rejected_frames,
        motor_port_opened=False,motor_serial_bytes_sent=0,motors_driven=False,
        calibration_verified=False,YOLO_deferred_by_user=True,power_issue_deferred_by_user=True)
    (folder/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return 0 if count and not receiver.decoder.crc_errors and not receiver.rejected_frames else 1


def run_cnn(args,config):
    if config.get('calibration_verified') is not True:
        raise ValueError('Actual wheel direction/speed calibration is not yet verified in vehicle_config.json')
    if not args.new_firmware_flashed:
        raise ValueError('New ULTDRV1 firmware must be installed before opening the vehicle port')
    name=choose_port(None)
    if name==args.motor_port:
        raise ValueError('FPGA and vehicle ports must differ')
    receiver=Receiver(json.loads((HERE.parent/'rpi_receiver/manifest.json').read_text(encoding='utf-8')))
    link=MotorLink(args.motor_port)
    try:
        with open_readonly(name,460800) as port:
            drain=time.monotonic()+2
            while time.monotonic()<drain:port.read(max(1,min(4096,port.in_waiting)))
            port.reset_input_buffer()
            started=time.monotonic();last_send=0.;latest=None;armed=False
            while time.monotonic()-started<args.seconds:
                data=port.read(max(1,min(4096,port.in_waiting)))
                now=time.monotonic()
                events=receiver.feed(data,now)
                for event in events:
                    if event['status']!='ready':
                        raise RuntimeError('Lane frame rejected; stop and restart manually')
                    latest=event
                if latest is None:
                    if now-started>2:raise TimeoutError('No valid fresh FPGA result')
                    continue
                if receiver.policy.stale(now):
                    raise TimeoutError('FPGA result stale; stopping')
                if not armed:
                    link.arm();armed=True
                if now-last_send>=.1:
                    target=wheel_targets(latest,config)
                    link.drive(target['left'],target['right']);last_send=time.monotonic()
    finally:
        link.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    commands=ap.add_subparsers(dest='action',required=True)
    p=commands.add_parser('preview');p.add_argument('--seconds',type=float,default=10);p.add_argument('--output',type=Path)
    p=commands.add_parser('bench',help='New firmware + lifted wheels only; no autonomous driving')
    p.add_argument('--motor-port',required=True);p.add_argument('--new-firmware-flashed',action='store_true')
    p.add_argument('--wheels-lifted',action='store_true')
    p.add_argument('--label',choices=['straight','left','right','sharp_left','sharp_right','slowdown'],default='straight')
    p.add_argument('--seconds',type=float,default=.5)
    p.add_argument('--pwm',type=int,help='Temporary bench PWM override, 1..120')
    p=commands.add_parser('run-cnn',help='Actual motion; requires completed physical calibration')
    p.add_argument('--motor-port',required=True);p.add_argument('--new-firmware-flashed',action='store_true')
    p.add_argument('--seconds',type=float,default=30)
    args=ap.parse_args()
    config=validate_config(json.loads((HERE/'vehicle_config.json').read_text(encoding='utf-8')))
    if args.action=='preview':
        if not 0<args.seconds<=3600:ap.error('Require 0 < seconds <= 3600')
        folder=args.output or HERE/'runs'/datetime.now(timezone.utc).strftime('preview_%Y%m%dT%H%M%SZ')
        return preview(args.seconds,folder,config)
    if args.action=='run-cnn':
        if not 0<args.seconds<=600:ap.error('Require 0 < seconds <= 600')
        run_cnn(args,config)
        return 0
    if not (args.new_firmware_flashed and args.wheels_lifted):
        ap.error('Bench requires confirmed new firmware and wheels lifted')
    if not 0<args.seconds<=2:ap.error('Bench duration must be > 0 and <= 2 seconds')
    if args.pwm is not None:
        if not 1<=args.pwm<=120:ap.error('PWM must be 1..120')
        config=dict(config,cruise_pwm=args.pwm,slow_pwm=min(config['slow_pwm'],args.pwm))
    label='straight' if args.label=='slowdown' else args.label
    event=dict(status='ready',inhibit_motion=False,steering_class=label,raw_class=label,
               slowdown_event=args.label=='slowdown')
    target=wheel_targets(event,config)
    link=MotorLink(args.motor_port)
    try:
        link.arm()
        until=time.monotonic()+args.seconds
        while time.monotonic()<until:
            link.drive(target['left'],target['right']);time.sleep(.05)
    finally:
        link.close()
    print('Bench pulse ended; stop sent. Actual wheel direction must be checked by the user.')
    return 0


if __name__=='__main__':raise SystemExit(main())
