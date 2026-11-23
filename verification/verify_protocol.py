"""Verify the extended protocol and integer hysteresis against the old policy."""
import dataclasses
import json
import math
import random
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from context_passage_policy import ContextPassagePolicy
from rpi_receiver import Frame,Decoder,SteeringPolicy,crc16,encode,SIZE


def main():
    m=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'));p=SteeringPolicy(m)
    def frame(i,q):return Frame(i,100,24000,p.model_id,2,1|(16 if q>=p.activate else 0)|(32 if q>=p.keep else 0),0x0201,30000,q)
    assert SIZE==36 and crc16(b'123456789')==0x29b1
    original=frame(0xffffffff,p.activate)
    packet=encode(original);decoder=Decoder()
    assert not decoder.feed(b'noise\xaa')
    frames=[]
    for byte in packet:frames+=decoder.feed(bytes([byte]))
    assert frames==[original]
    corrupt=bytearray(packet);corrupt[12]^=1
    assert decoder.feed(bytes(corrupt)+packet)==[original] and decoder.crc_errors==1
    assert p.update(original,0)['steering_class']=='straight'
    assert p.update(frame(0,p.keep),.1)['crossing_context_active']
    # Continuous fresh packets with a low context score allow grace to expire.
    for i in range(1,16):assert p.update(frame(i,p.keep-100),.1+i*.09)['crossing_context_active']
    assert not p.update(frame(16,p.keep-100),1.61)['crossing_context_active']
    for bad in (frame(16,0),dataclasses.replace(frame(17,0),model_id=0),
                dataclasses.replace(frame(17,0),flags=3),dataclasses.replace(frame(17,0),preprocess_id=0)):
        try:p.update(bad,1.62)
        except ValueError:pass
        else:raise AssertionError('Bad packet accepted')
    assert p.stale(2)
    # Idle gaps must clear a previous passage, avoiding carryover on reconnect.
    p.reset();p.update(frame(1,p.activate),0)
    assert not p.update(frame(2,0),.5)['crossing_context_active']
    # Compare the integer port against the actual existing Webots policy.
    p.reset();old=ContextPassagePolicy(m['thresholds']['activate_probability'])
    rng=random.Random(20261025)
    scores=[p.activate-1,p.activate,p.keep-1,p.keep,0]+[rng.randrange(-200000,200000) for _ in range(500)]
    for i,q in enumerate(scores):
        now=i*.08
        expected=old.choose(1/(1+math.exp(-q/(1<<m['thresholds']['fractional_bits']))),'right',now)
        assert p.update(frame(i,q),now)['steering_class']==expected
    (HERE/'sample_uart_packet.bin').write_bytes(packet)
    (HERE/'expected_uart_packet.hex').write_text(''.join(f'{b:02X}\n' for b in packet),encoding='ascii')
    report=dict(status='passed',packet_bytes=SIZE,crc_known_vector='29B1',
        verified=['fragmented input','noise resynchronization','CRC corruption','frame-id wrap',
                  'duplicate/out-of-order rejection','wrong model/preprocessing rejection',
                  'error flags','activation and keep thresholds','1.5s grace release','stale reset',
                  '505-step equivalence to the original temporal policy at quantized scores'])
    (HERE/'protocol_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
