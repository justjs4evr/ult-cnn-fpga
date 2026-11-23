"""One supervised ground-load pulse; does not enable calibrated autonomous driving."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from vehicle_control import HERE, MotorLink
from auto_drive import auto_motor_port
from drive_protocol import validate_config, wheel_targets


def validate_trial(config, seconds, clear_floor, motor_power_on, long_candidate=False):
    validate_config(config)
    if not clear_floor or not motor_power_on:
        raise ValueError('Confirm clear floor and motor power before the supervised pulse')
    if config.get('lifted_wheel_mapping_verified') is not True:
        raise ValueError('Actual lifted-wheel mapping must already be verified')
    limit=10 if long_candidate else 1
    if not math.isfinite(seconds) or not 0 < seconds <= limit:
        raise ValueError(f'Ground calibration pulse must be positive and at most {limit} seconds')
    if max(config['cruise_pwm'], config['slow_pwm'], config.get('turn_pwm',0)) > 120:
        raise ValueError('Ground calibration pulse is limited to PWM120')


def pulse(link, target, seconds, stop_file):
    if stop_file.exists():
        raise RuntimeError('AUTO_STOP is present')
    link.arm()
    deadline = time.monotonic() + seconds
    commands = 0
    while time.monotonic() < deadline:
        if stop_file.exists():
            raise RuntimeError('User stop requested')
        link.drive(target['left'], target['right'])
        commands += 1
        time.sleep(min(.05, max(0, deadline - time.monotonic())))
    return commands


def run(label, seconds, clear_floor, motor_power_on, motor_port=None, output=None, config_path=None, long_candidate=False):
    config = json.loads((config_path or HERE / 'vehicle_config.json').read_text(encoding='utf-8'))
    validate_trial(config, seconds, clear_floor, motor_power_on, long_candidate)
    if long_candidate and config_path is None:
        raise ValueError('Long supervised run requires a separate candidate config')
    if label not in ('straight', 'slowdown', 'left', 'right', 'sharp_left', 'sharp_right'):
        raise ValueError('Unknown ground calibration maneuver')
    stop_file = HERE / 'AUTO_STOP'
    if stop_file.exists():
        raise RuntimeError('AUTO_STOP is present; clear deliberately before a new trial')
    target = wheel_targets(dict(status='ready', inhibit_motion=False,
        steering_class='straight' if label == 'slowdown' else label,
        raw_class='straight' if label == 'slowdown' else label,
        slowdown_event=label == 'slowdown'), config)
    port = auto_motor_port(motor_port)
    folder = output or HERE / 'runs' / datetime.now(timezone.utc).strftime('ground_%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    link = None
    error = None
    stop_ack = False
    commands = 0
    started = time.monotonic()
    try:
        link = MotorLink(port)
        if link.exchange('C') != ['C', str(link.seq), 'W4']:
            raise RuntimeError('Expected W4 vehicle firmware')
        commands = pulse(link, target, seconds, stop_file)
    except KeyboardInterrupt:
        error = 'Interrupted by user'
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if link is not None:
            try:
                link.close()
                stop_ack = True
            except Exception as exc:
                error = (error + '; ' if error else '') + 'STOP ACK unavailable: ' + str(exc)
    report = dict(status='passed' if error is None and commands and stop_ack else 'failed',
        timestamp_utc=datetime.now(timezone.utc).isoformat(), mode='supervised_ground_pulse',
        label=label, requested_pulse_seconds=seconds, target=target, motor_port=port,
        command_ACK_count=commands, STOP_ACK_verified=stop_ack, error=error,
        total_elapsed_seconds=time.monotonic()-started,
        clear_floor_confirmed=clear_floor, motor_power_on_confirmed=motor_power_on,
        physical_observation=None, ground_calibration_completed=False,
        autonomous_driving=False, camera_or_YOLO_used=False, config=config)
    (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(folder=str(folder), **report), ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'passed' else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--label', choices=['straight', 'slowdown', 'left', 'right', 'sharp_left', 'sharp_right'], default='straight')
    ap.add_argument('--seconds', type=float, default=.5)
    ap.add_argument('--clear-floor', action='store_true')
    ap.add_argument('--motor-power-on', action='store_true')
    ap.add_argument('--motor-port')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--config', type=Path, help='Separate calibration candidate; does not replace active settings')
    ap.add_argument('--long-candidate', action='store_true', help='Allow up to 10 seconds only with --config')
    args = ap.parse_args()
    return run(args.label, args.seconds, args.clear_floor, args.motor_power_on, args.motor_port, args.output,args.config,args.long_candidate)


if __name__ == '__main__':
    raise SystemExit(main())
