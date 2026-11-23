"""Portable handoff launcher. No arguments opens a menu; no automatic motion."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
VEHICLE = ROOT / 'vehicle_control'
sys.path.insert(0, str(VEHICLE))
from drive_protocol import validate_config, wheel_targets


def load_settings(path=ROOT/'DRIVE_SETTINGS.json'):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    for key in ('straight_pwm', 'slow_pwm', 'gentle_outer_pwm', 'gentle_inner_pwm',
                'sharp_outer_pwm', 'sharp_inner_pwm'):
        if type(value.get(key)) is not int or not 1 <= value[key] <= 120:
            raise ValueError(key+' must be an integer in 1..120')
    if value['slow_pwm'] > value['straight_pwm']:
        raise ValueError('slow_pwm must not exceed straight_pwm')
    if not 0 < value['sharp_inner_pwm'] < value['sharp_outer_pwm']:
        raise ValueError('Sharp inner must be below sharp outer')
    if not 0 < value['gentle_inner_pwm'] < value['gentle_outer_pwm']:
        raise ValueError('Gentle inner must be below gentle outer')
    # Find an integer percent that reproduces the requested PWM exactly.
    def ratio(inner, outer):
        matches = [r for r in range(101) if outer*r//100 == inner]
        if not matches:
            raise ValueError('Requested pair cannot be represented; choose a nearby inner PWM')
        return matches[0]
    gentle_ratio = ratio(value['gentle_inner_pwm'], value['gentle_outer_pwm'])
    sharp_ratio = ratio(value['sharp_inner_pwm'], value['sharp_outer_pwm'])
    if sharp_ratio >= gentle_ratio:
        raise ValueError('Sharp turn must have a lower inner/outer ratio than gentle turn')
    profile = dict(protocol='ULTDRV1', wheel_type='mecanum',
        cruise_pwm=value['straight_pwm'], slow_pwm=value['slow_pwm'],
        turn_pwm=value['gentle_outer_pwm'], gentle_turn_pwm=value['gentle_outer_pwm'],
        sharp_turn_pwm=value['sharp_outer_pwm'], slow_gentle_turn_pwm=value['gentle_outer_pwm'],
        slow_sharp_turn_pwm=value['sharp_outer_pwm'], gentle_inner_percent=gentle_ratio,
        sharp_inner_percent=sharp_ratio, inner_pwm_floor=min(value['gentle_inner_pwm'], value['sharp_inner_pwm']),
        calibration_verified=False, lifted_wheel_mapping_verified=True, ground_load_test_verified=False,
        note='Portable handoff profile; supervised tests only. No reverse, timed decay, or lateral motion.')
    validate_config(profile)
    return value, profile


def target_table(profile):
    rows = []
    for label in ('straight', 'left', 'right', 'sharp_left', 'sharp_right', 'stop'):
        event = dict(status='ready', inhibit_motion=False, steering_class=label,
                     raw_class=label, slowdown_event=label == 'stop')
        target = wheel_targets(event, profile)
        rows.append(dict(label=label, left=target['left'], right=target['right']))
    return rows


def write_profile(profile):
    target = VEHICLE / 'handoff_active_profile.json'
    target.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding='utf-8')
    return target


def action(args):
    if args.action == 'stop':
        (VEHICLE/'AUTO_STOP').write_text('Handoff user stop requested\n', encoding='ascii')
        print('정지를 요청했습니다. 실행 중인 프로그램이 이를 확인해 정지합니다.')
        return 0
    if args.action == 'clear-stop':
        if not args.motor_power_off:
            raise ValueError('모터 전원을 끈 뒤 --motor-power-off를 지정하세요.')
        (VEHICLE/'AUTO_STOP').unlink(missing_ok=True)
        print('정지 잠금만 해제했습니다. 차량은 출발하지 않습니다.')
        return 0
    if args.action == 'ports':
        from vehicle_control import Receiver  # makes bundled pyserial available
        from serial.tools import list_ports
        for port in list_ports.comports():
            print(json.dumps(dict(port=port.device, description=port.description,
                                  vid=port.vid, pid=port.pid, serial=port.serial_number), ensure_ascii=False))
        return 0
    settings, profile = load_settings(args.settings)
    if args.action == 'check':
        print(json.dumps(dict(settings=settings, outputs=target_table(profile),
                              motor_port_opened=False, motors_driven=False), ensure_ascii=False, indent=2))
        return 0
    fpga = args.fpga_port or settings.get('fpga_port') or None
    motor = args.motor_port or settings.get('motor_port') or None
    if args.action == 'preview':
        if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
            raise ValueError('Preview seconds must be in 0..60')
        from vehicle_control import preview
        folder = VEHICLE/'runs'/datetime.now(timezone.utc).strftime('handoff_preview_%Y%m%dT%H%M%S_%fZ')
        return preview(args.seconds, folder, profile, fpga_port=fpga)
    # Readiness flags are intentionally not saved to DRIVE_SETTINGS.json.
    if not (args.clear_floor and args.motor_power_on):
        raise ValueError('실제 구동에는 --clear-floor --motor-power-on이 모두 필요합니다.')
    profile_path = write_profile(profile)
    if args.action == 'drive':
        from supervised_cnn_ground import run
        return run(args.seconds, True, True, motor_port=motor,
                   config_path=profile_path, fpga_port=fpga)
    if args.action == 'fixed':
        from ground_trial import run
        return run(args.label, args.seconds, True, True, motor_port=motor,
                   config_path=profile_path, long_candidate=False)
    raise ValueError('Unknown action')


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['check','ports','preview','fixed','drive','stop','clear-stop'])
    ap.add_argument('--settings', type=Path, default=ROOT/'DRIVE_SETTINGS.json')
    ap.add_argument('--seconds', type=float, default=5)
    ap.add_argument('--label', choices=['straight','slowdown','left','right','sharp_left','sharp_right'], default='straight')
    ap.add_argument('--motor-port')
    ap.add_argument('--fpga-port')
    ap.add_argument('--clear-floor', action='store_true')
    ap.add_argument('--motor-power-on', action='store_true')
    ap.add_argument('--motor-power-off', action='store_true')
    return ap


def menu():
    while True:
        print('\nCNN 차량 실행 메뉴\n1 설정·출력 확인  2 설정 파일 위치\n3 USB 포트 목록  4 CNN 출력만 확인(모터 구동 없음)\n5 고정 방향 0.5초 시험  6 CNN 5초 시험  7 CNN 10초 시험\n8 정지 요청  9 정지 잠금 해제(모터 전원 OFF)  0 종료')
        choice = input('번호: ').strip()
        if choice == '0':
            return 0
        if choice == '2':
            print('이 파일의 숫자 여섯 개를 수정하세요:', ROOT/'DRIVE_SETTINGS.json')
            continue
        argv = {'1':['check'], '3':['ports'], '4':['preview'], '8':['stop']}.get(choice)
        if choice in ('5','6','7'):
            if input('차량 주변·배선 여유를 확인하고 모터 전원을 켰으면 READY 입력: ').strip() != 'READY':
                continue
            if choice == '5':
                label = input('straight / slowdown / left / right / sharp_left / sharp_right: ').strip()
                argv = ['fixed','--seconds','.5','--label',label]
            else:
                argv = ['drive','--seconds','5' if choice == '6' else '10']
            argv += ['--clear-floor','--motor-power-on']
        if choice == '9':
            if input('모터 전원을 껐으면 OFF 입력: ').strip() == 'OFF':
                argv = ['clear-stop','--motor-power-off']
        if not argv:
            continue
        try:
            action(parser().parse_args(argv))
        except (Exception, SystemExit) as exc:
            print('오류:', exc)


if __name__ == '__main__':
    try:
        raise SystemExit(action(parser().parse_args()) if len(sys.argv)>1 else menu())
    except Exception as exc:
        print('오류:', exc)
        raise SystemExit(1)
