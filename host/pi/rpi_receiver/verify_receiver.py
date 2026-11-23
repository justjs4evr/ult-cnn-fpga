"""Exercise corruption, semantic policy, invalid frames and timeout behavior."""
from dataclasses import replace
import json
from pathlib import Path
from protocol import Frame, encode, crc16
from receiver import Receiver

HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
    r = Receiver(manifest)
    mid = r.policy.model_id
    def frame(i, raw=0, q=0, tick=None):
        return Frame(i, i*100 if tick is None else tick, 31611, mid, raw,
            1 | (16 if q >= 5810 else 0) | (32 if q >= 4500 else 0), 0x0201, 128000, q)
    assert crc16(b'123456789') == 0x29b1
    packet = encode(frame(1))
    events = []
    for part in [b'noise\xaa', packet[:3], packet[3:9], packet[9:]]:
        events += r.feed(part, 0)
    assert len(events) == 1 and events[-1]['steering_class'] == 'straight'
    assert not events[-1]['motors_enabled']
    event = r.feed(encode(frame(2, raw=5)), .1)[-1]
    assert event['slowdown_event'] and event['steering_class'] == 'straight' and not event['stop_required']
    event = r.feed(encode(frame(3, raw=2, q=5810)), .2)[-1]
    assert event['crossing_context_active'] and event['steering_class'] == 'right'
    for i in range(4, 19):
        assert r.feed(encode(frame(i, raw=2)), .2+(i-3)*.09)[-1]['crossing_context_active']
    assert r.feed(encode(frame(19, raw=2)), 1.81)[-1]['steering_class'] == 'right'
    assert r.poll(2.6)['status'] == 'stale'
    assert r.policy.stale(2.6)
    cases = [replace(frame(22), model_id=0), replace(frame(22), preprocess_id=0),
             replace(frame(22), flags=3), replace(frame(22), flags=0x81),
             replace(frame(22), raw_class=6), replace(frame(22), capture_to_tx_us=250001),
             replace(frame(22), context_logit_q=5810)]
    for bad in cases:
        r = Receiver(manifest)
        event = r.feed(encode(bad), 0)[-1]
        assert event['status'] == 'rejected' and event['inhibit_motion']
    for bad in [frame(21), frame(20), frame(22, tick=2100)]:
        r = Receiver(manifest)
        r.feed(encode(frame(21)), 0)
        assert r.feed(encode(bad), .1)[-1]['status'] == 'rejected'
    r = Receiver(manifest)
    r.feed(encode(frame(0xffffffff, tick=0xfffffffe)), 0)
    assert r.feed(encode(frame(0, tick=10)), .1)[-1]['status'] == 'ready'
    corrupt = bytearray(packet)
    corrupt[12] ^= 1
    events = Receiver(manifest).feed(bytes(corrupt) + packet, 0)
    assert [e['status'] for e in events] == ['rejected', 'ready']
    out = dict(status='passed', checks=['fragmentation/noise/CRC recovery', 'six-class validation',
        'STOP slowdown without full stop', 'context activation and 1.5s hold', 'stale inhibit',
        'model/preprocess/flag/age rejection', 'duplicate/order/tick rejection', 'counter wrap'],
        motors_driven=False, hardware_used=False)
    (HERE / 'software_verification.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
