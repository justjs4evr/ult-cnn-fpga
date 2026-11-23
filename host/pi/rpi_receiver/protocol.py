"""Protocol v2 decoder and existing crossing policy. Does not drive GPIO/motors.

Camera A/YOLO and speed/steering arbitration remain in the Raspberry Pi app.
Frames from camera B/FPGA carry raw six-class output plus context logit.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import time

HERE=Path(__file__).resolve().parent
CLASSES=('straight','left','right','sharp_left','sharp_right','stop')
BODY=struct.Struct('<2sBBIIIIBBH IiH'.replace(' ',''))
SIZE=BODY.size+2  # 36 bytes; v1's 32-byte packets are intentionally rejected.


def crc16(data):
    crc=0xffff
    for byte in data:
        crc^=byte<<8
        for _ in range(8):crc=((crc<<1)^0x1021)&0xffff if crc&0x8000 else (crc<<1)&0xffff
    return crc


@dataclass
class Frame:
    frame_id:int
    capture_tick:int
    inference_us:int
    model_id:int
    raw_class:int
    flags:int
    preprocess_id:int
    capture_to_tx_us:int
    context_logit_q:int


def encode(frame):
    body=BODY.pack(b'\xaa\x55',2,1,frame.frame_id,frame.capture_tick,frame.inference_us,frame.model_id,
                   frame.raw_class,frame.flags,frame.preprocess_id,frame.capture_to_tx_us,frame.context_logit_q,0)
    return body+struct.pack('<H',crc16(body))


class Decoder:
    def __init__(self):self.buffer=bytearray();self.crc_errors=0
    def feed(self,data):
        self.buffer.extend(data);frames=[]
        while True:
            where=self.buffer.find(b'\xaa\x55')
            if where<0:
                self.buffer[:]=self.buffer[-1:] if self.buffer[-1:]==b'\xaa' else b''
                break
            if where:del self.buffer[:where]
            if len(self.buffer)<4:break
            if self.buffer[2:4]!=b'\x02\x01':del self.buffer[0];continue
            if len(self.buffer)<SIZE:break
            packet=bytes(self.buffer[:SIZE])
            if crc16(packet[:-2])!=struct.unpack('<H',packet[-2:])[0]:
                self.crc_errors+=1;del self.buffer[0];continue
            values=BODY.unpack(packet[:-2])
            if values[-1]!=0:del self.buffer[0];continue
            del self.buffer[:SIZE]
            frames.append(Frame(*values[3:-1]))
        return frames


class SteeringPolicy:
    def __init__(self,manifest,stale_seconds=.30):
        self.activate=manifest['thresholds']['activate_q'];self.keep=manifest['thresholds']['keep_q']
        self.clear=manifest['thresholds']['clear_seconds'];self.stale_seconds=stale_seconds
        self.model_id=int.from_bytes(bytes.fromhex(manifest['checkpoint_sha256'])[:4],'little')
        self.preprocess_id=0x0201;self.reset()
    def reset(self):
        self.active=False;self.last_keep=-float('inf');self.last_frame=None;self.received_at=None
    def update(self,frame,now=None):
        now=time.monotonic() if now is None else now
        if self.received_at is not None and now-self.received_at>=self.stale_seconds:self.reset()
        if frame.model_id!=self.model_id or frame.preprocess_id!=self.preprocess_id:
            raise ValueError('Model/preprocessing version mismatch')
        if not frame.flags&1 or frame.flags&0xce or not 0<=frame.raw_class<6:
            raise ValueError('Invalid frame or camera/overflow/saturation error')
        if frame.capture_to_tx_us>250000:raise ValueError('Frame too old inside FPGA')
        activate=frame.context_logit_q>=self.activate;keep=frame.context_logit_q>=self.keep
        if bool(frame.flags&16)!=activate or bool(frame.flags&32)!=keep:
            raise ValueError('Context flag/logit mismatch')
        if self.last_frame is not None:
            delta=(frame.frame_id-self.last_frame)&0xffffffff
            if not 0<delta<0x80000000:raise ValueError('Duplicate or out-of-order frame')
        self.last_frame=frame.frame_id;self.received_at=now
        if activate:self.active=True;self.last_keep=now
        elif self.active and keep:self.last_keep=now
        elif self.active and now-self.last_keep>=self.clear:self.active=False
        raw=CLASSES[frame.raw_class]
        # Crossing/crosswalk context reduces speed but does not erase steering.
        # A raw STOP class is the only class that maps to straight slowdown.
        steering='straight' if raw=='stop' else raw
        return dict(frame_id=frame.frame_id,raw_class=raw,steering_class=steering,
                    crossing_context_active=self.active,slowdown_event=(self.active or raw=='stop'),
                    context_logit_q=frame.context_logit_q)
    def stale(self,now=None):
        now=time.monotonic() if now is None else now
        return self.received_at is None or now-self.received_at>=self.stale_seconds


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port',help='Optional serial port; requires pyserial')
    ap.add_argument('--baud',type=int,default=460800)
    ap.add_argument('--packet-file',type=Path,help='Replay binary UART bytes without hardware')
    ap.add_argument('--manifest',type=Path,default=HERE/'manifest.json')
    args=ap.parse_args();manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    decoder=Decoder();policy=SteeringPolicy(manifest)
    def consume(data):
        for frame in decoder.feed(data):
            try:print(json.dumps(policy.update(frame)),flush=True)
            except ValueError as exc:print(json.dumps({'rejected':str(exc)}),flush=True)
    if args.packet_file:consume(args.packet_file.read_bytes())
    elif args.port:
        try:import serial
        except ImportError:raise SystemExit('Install pyserial in the Raspberry Pi environment: python3 -m pip install pyserial')
        with serial.Serial(args.port,args.baud,timeout=.05) as port:
            while True:
                consume(port.read(4096))
                if policy.stale():
                    print(json.dumps({'status':'stale','steering_class':None}),flush=True)
                    policy.reset()
    else:ap.error('Specify --packet-file or --port')


if __name__=='__main__':main()
