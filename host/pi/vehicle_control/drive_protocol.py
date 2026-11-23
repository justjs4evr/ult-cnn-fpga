"""ULTDRV1 bounded ASCII frames; CRC16-CCITT-FALSE over the body."""
from dataclasses import dataclass


def crc16(data):
    crc=0xffff
    for byte in data:
        crc ^= byte<<8
        for _ in range(8):
            crc=((crc<<1)^0x1021)&0xffff if crc&0x8000 else (crc<<1)&0xffff
    return crc


def encode(*fields):
    body=','.join(str(x) for x in fields).encode('ascii')
    if len(body)>55 or any(x in body for x in (b'$',b'#',b'*',b'\n',b'\r')):
        raise ValueError('Invalid body')
    return b'$'+body+f'*{crc16(body):04X}#'.encode('ascii')


class Decoder:
    def __init__(self):
        self.buffer=bytearray(); self.errors=0
    def feed(self,data):
        frames=[]
        for byte in data:
            if byte==36:
                self.buffer=bytearray(b'$')
            elif self.buffer:
                self.buffer.append(byte)
                if len(self.buffer)>64:
                    self.errors+=1; self.buffer.clear()
                elif byte==35:
                    packet=bytes(self.buffer); self.buffer.clear()
                    try:
                        body,check=packet[1:-1].rsplit(b'*',1)
                        if len(check)!=4 or int(check,16)!=crc16(body):
                            raise ValueError('CRC')
                        frames.append(body.decode('ascii').split(','))
                    except (ValueError,UnicodeError):
                        self.errors+=1
        return frames


def wheel_targets(event,config):
    """Return proposed PWM targets only; STOP class is slowdown, not a halt."""
    if event.get('status')!='ready' or event.get('inhibit_motion',True):
        return dict(left=0,right=0,reason='no_valid_lane_result')
    label=event.get('steering_class')
    if label=='stop':
        label='straight'
    if label not in ('straight','left','right','sharp_left','sharp_right'):
        return dict(left=0,right=0,reason='unknown_class')
    slowdown=bool(event.get('slowdown_event')) or event.get('raw_class')=='stop'
    if label.startswith('sharp') and config.get('sharp_pivot_pwm') is not None:
        pivot=config['sharp_pivot_pwm']
        left,right=(-pivot,pivot) if label.endswith('left') else (pivot,-pivot)
        return dict(left=left,right=right,reason='slowdown' if slowdown else label)
    pwm=config['slow_pwm'] if slowdown else config['cruise_pwm']
    if label!='straight':
        turn_key='sharp_turn_pwm' if label.startswith('sharp') else 'gentle_turn_pwm'
        if slowdown:
            # Ground-loaded skid steering needs its own reduced turn output;
            # scaling straight slow_pwm by the inner ratio can stall the car.
            pwm=config.get('slow_'+turn_key,pwm)
        else:
            pwm=config.get(turn_key,config.get('turn_pwm',pwm))
    if label=='straight':
        left=right=pwm
    else:
        ratio=config['sharp_inner_percent'] if label.startswith('sharp') else config['gentle_inner_percent']
        inner=max(pwm*ratio//100,config.get('inner_pwm_floor',0))
        left,right=(inner,pwm) if label.endswith('left') else (pwm,inner)
    return dict(left=left,right=right,reason='slowdown' if slowdown else label)


def validate_config(config):
    if not 0<config['slow_pwm']<=config['cruise_pwm']<=120:
        raise ValueError('Require 0 < slow_pwm <= cruise_pwm <= 120')
    if type(config.get('turn_pwm',config['cruise_pwm'])) is not int or not 0<config.get('turn_pwm',config['cruise_pwm'])<=120:
        raise ValueError('Turn PWM must be an integer in 1..120')
    for key in ('gentle_turn_pwm','sharp_turn_pwm'):
        if key in config and (type(config[key]) is not int or not 0<config[key]<=120):
            raise ValueError(f'{key} must be an integer in 1..120')
        slow_key='slow_'+key
        if slow_key in config:
            normal=config.get(key,config.get('turn_pwm',config['cruise_pwm']))
            if type(config[slow_key]) is not int or not 0<config[slow_key]<=normal:
                raise ValueError(f'{slow_key} must be an integer in 1..{normal}')
    if not 0<=config['sharp_inner_percent']<config['gentle_inner_percent']<=100:
        raise ValueError('Invalid inner-wheel ratios')
    if 'inner_pwm_floor' in config:
        floor=config['inner_pwm_floor']
        turn_pwms=[config.get('turn_pwm',config['cruise_pwm']),config['slow_pwm']]
        turn_pwms.extend(config[k] for k in ('gentle_turn_pwm','sharp_turn_pwm','slow_gentle_turn_pwm','slow_sharp_turn_pwm') if k in config)
        if type(floor) is not int or not 0<=floor<min(turn_pwms):
            raise ValueError('Inner PWM floor must be a nonnegative integer below every turn output')
        if 'sharp_pivot_pwm' in config:
            raise ValueError('Positive inner PWM floor cannot be combined with a reverse pivot profile')
    if 'sharp_pivot_pwm' in config:
        if config.get('wheel_type')!='mecanum' or type(config['sharp_pivot_pwm']) is not int or not 0<config['sharp_pivot_pwm']<=45:
            raise ValueError('Mecanum pivot trial requires integer PWM1..45')
        limit=config.get('sharp_pivot_max_seconds')
        if type(limit) not in (int,float) or not 0<limit<=.5:
            raise ValueError('Mecanum pivot trial requires a duration limit <=0.5 seconds')
    if 'sharp_turn_decay' in config:
        d=config['sharp_turn_decay']
        if not isinstance(d,dict) or 'sharp_pivot_pwm' in config:
            raise ValueError('Turn decay requires forward-only steering')
        for key,lo,hi in (('boost_seconds',.1,1.),('reset_seconds',.3,2.)):
            value=d.get(key)
            if type(value) not in (int,float) or not lo<=value<=hi:
                raise ValueError(f'Invalid turn decay {key}')
        if type(d.get('reset_frames')) is not int or not 2<=d['reset_frames']<=10:
            raise ValueError('Turn decay reset_frames must be2..10')
        outer=d.get('relaxed_outer_pwm');inner=d.get('relaxed_inner_pwm')
        max_outer=min(config.get('sharp_turn_pwm',config.get('turn_pwm',config['cruise_pwm'])),
                      config.get('slow_sharp_turn_pwm',config['slow_pwm']))
        if type(outer) is not int or type(inner) is not int or not max(1,config.get('inner_pwm_floor',0))<=inner<outer<=max_outer:
            raise ValueError('Invalid relaxed steering PWM values')
    return config
