"""One-wheel lifted bench pulse; logs the requested logical channel group."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
from vehicle_control import MotorLink

HERE=Path(__file__).resolve().parent
WHEELS=['front_left','rear_left','front_right','rear_right']
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--port',required=True)
ap.add_argument('--wheel',choices=WHEELS,required=True)
ap.add_argument('--pwm',type=int,default=50)
ap.add_argument('--seconds',type=float,default=.4)
ap.add_argument('--wheels-lifted',action='store_true')
args=ap.parse_args()
if not args.wheels_lifted:ap.error('Lifted wheels confirmation required')
if not 1<=args.pwm<=120 or not 0<args.seconds<=.5:ap.error('Require PWM 1..120, duration 0..0.5 s')
targets=[0,0,0,0];targets[WHEELS.index(args.wheel)]=args.pwm
folder=HERE/'runs'/datetime.now(timezone.utc).strftime('wheel_%Y%m%dT%H%M%S_%fZ')
folder.mkdir(parents=True)
report=dict(wheel=args.wheel,targets_LF_LR_RF_RR=targets,requested_pulse_seconds=args.seconds,
    wheels_lifted_confirmed_by_user=True,physical_observation=None,protocol_completed=False)
link=None
try:
    link=MotorLink(args.port)
    if link.exchange('C')!=['C',str(link.seq),'W4']:raise RuntimeError('W4 capability absent')
    link.arm()
    until=time.monotonic()+args.seconds
    while time.monotonic()<until:
        link.drive_wheels(targets);time.sleep(.05)
    report['protocol_completed']=True
finally:
    try:
        if link is not None:link.close()
        report['stop_ACK_verified']=link is not None
    finally:
        (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(folder=str(folder),**report),ensure_ascii=False,indent=2))
