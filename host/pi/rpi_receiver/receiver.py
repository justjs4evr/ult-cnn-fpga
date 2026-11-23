"""Read-only FPGA decision receiver, portable between Windows and Raspberry Pi.

No serial writes, GPIO, motor driver or YOLO are implemented in this stage.
The capture tick is XIAO milliseconds, not the Pi clock. It cannot by itself
provide absolute image age. USB buffers are flushed on connection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
from protocol import Decoder, SteeringPolicy


class Receiver:
    def __init__(self, manifest, stale_seconds=.75):
        if not math.isfinite(stale_seconds) or stale_seconds <= 0:
            raise ValueError('stale_seconds must be finite and positive')
        self.decoder = Decoder()
        self.policy = SteeringPolicy(manifest, stale_seconds)
        self.last_tick = None
        self.last_status = None
        self.valid_frames = 0
        self.rejected_frames = 0
        self.intervals = []
        self.last_receive = None
        self.sequence = 0

    def event(self, status, **values):
        self.last_status = status
        self.sequence += 1
        return dict(sequence=self.sequence, status=status, motors_enabled=False,
                    **values)

    def inhibit(self, status, reason):
        self.policy.reset()
        self.last_tick = None
        self.last_receive = None
        return self.event(status, steering_class=None, slowdown_event=False,
                          inhibit_motion=True, reason=reason)

    def poll(self, now):
        if self.policy.stale(now) and self.last_status not in ('waiting', 'stale', 'rejected', 'disconnected'):
            return self.inhibit('stale', 'No fresh valid FPGA result within timeout')
        return None

    def feed(self, data, now):
        events = []
        if self.policy.stale(now) and self.policy.received_at is not None:
            self.policy.reset()
            self.last_tick = None
            self.last_receive = None
        old_crc = self.decoder.crc_errors
        frames = self.decoder.feed(data)
        if self.decoder.crc_errors != old_crc:
            events.append(self.inhibit('rejected', 'CRC error; waiting for a new valid frame'))
        for frame in frames:
            try:
                if self.last_tick is not None:
                    delta = (frame.capture_tick - self.last_tick) & 0xffffffff
                    if not 0 < delta < 0x80000000:
                        raise ValueError('Capture tick did not advance')
                decision = self.policy.update(frame, now)
            except ValueError as exc:
                self.rejected_frames += 1
                events.append(self.inhibit('rejected', str(exc)))
                continue
            if self.last_receive is not None and now > self.last_receive:
                self.intervals.append(now - self.last_receive)
            self.last_receive = now
            self.last_tick = frame.capture_tick
            self.valid_frames += 1
            events.append(self.event('ready', **decision, inhibit_motion=False,
                capture_tick_ms=frame.capture_tick, inference_us=frame.inference_us,
                fpga_input_to_tx_us=frame.capture_to_tx_us,
                # User policy: crossroads/crosswalk mean slowdown, not a stop.
                stop_required=False, absolute_camera_age_verified=False))
        return events


def ports():
    from serial.tools import list_ports
    return list(list_ports.comports())


def choose_port(explicit):
    if explicit:
        return explicit
    # Linux exposes one shared serial number and distinct interface aliases;
    # Windows appends A/B to that same device serial number.
    alias = Path('/dev/serial/by-id/usb-SIPEED_USB_Debugger_2025030317-if01-port0')
    if platform.system() == 'Linux' and alias.exists():
        return str(alias)
    candidates = [p.device for p in ports() if (p.vid, p.pid) == (0x0403, 0x6010)
                  and p.serial_number == '2025030317B']
    if len(candidates) != 1:
        raise ValueError('Cannot identify FPGA result UART. Run --list-ports and use --port.')
    return candidates[0]


def open_readonly(name, baud):
    import serial
    options = dict(port=None, baudrate=baud, timeout=.05, xonxoff=False,
                   rtscts=False, dsrdtr=False)
    if platform.system() != 'Windows':
        options['exclusive'] = True
    port = serial.Serial(**options)
    port.dtr = False
    port.rts = False
    port.port = name
    port.open()
    port.reset_input_buffer()
    return port


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--list-ports', action='store_true')
    ap.add_argument('--port')
    ap.add_argument('--baud', type=int, default=460800)
    ap.add_argument('--seconds', type=float, default=0, help='0 runs until Ctrl+C')
    ap.add_argument('--stale-seconds', type=float, default=.75)
    ap.add_argument('--warmup-seconds', type=float, default=2,
                    help='Drain startup USB buffers without issuing decisions')
    ap.add_argument('--quiet', action='store_true', help='Save events without printing each frame')
    ap.add_argument('--manifest', type=Path, default=HERE / 'manifest.json')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.list_ports:
        for p in ports():
            print(json.dumps(dict(port=p.device, description=p.description,
                                  serial_number=p.serial_number, vid=p.vid, pid=p.pid,
                                  interface=p.interface, location=p.location), ensure_ascii=False))
        return 0
    if not math.isfinite(args.seconds) or args.seconds < 0:
        ap.error('--seconds must be finite and nonnegative')
    if not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 2:
        ap.error('--warmup-seconds must be finite and at least 2')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    receiver = Receiver(manifest, args.stale_seconds)
    name = choose_port(args.port)
    # UTC is used only for machine logs and portable folder naming.
    folder = args.output or HERE / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    result = 'completed'
    error = None
    ready = 0
    startup_decoder = Decoder()
    startup_frames = 0
    with (folder / 'events.jsonl').open('w', encoding='utf-8') as log, (folder / 'received.bin').open('wb') as raw:
        def emit(event):
            event['receiver_elapsed_seconds'] = time.monotonic() - start
            line = json.dumps(event, ensure_ascii=False)
            if not args.quiet:
                print(line, flush=True)
            log.write(line + '\n')
            log.flush()
        emit(receiver.inhibit('waiting', 'Waiting for FPGA autonomous results'))
        try:
            with open_readonly(name, args.baud) as port:
                drain_until = time.monotonic() + args.warmup_seconds
                while time.monotonic() < drain_until:
                    data = port.read(max(1, min(4096, port.in_waiting)))
                    raw.write(data)
                    startup_frames += len(startup_decoder.feed(data))
                port.reset_input_buffer()
                emit(receiver.inhibit('waiting', 'Startup drain complete; waiting for new FPGA results'))
                receive_start = time.monotonic()
                while args.seconds == 0 or time.monotonic() - receive_start < args.seconds:
                    data = port.read(max(1, min(4096, port.in_waiting)))
                    now = time.monotonic()
                    raw.write(data)
                    events = receiver.feed(data, now)
                    # Keep all events in the audit log. A vehicle adapter must use
                    # only the latest event, rather than queue old steering commands.
                    for event in events:
                        ready += event['status'] == 'ready'
                        emit(event)
                    stale = receiver.poll(now)
                    if stale:
                        emit(stale)
        except KeyboardInterrupt:
            result = 'interrupted'
        except (OSError, ValueError) as exc:
            error = str(exc)
            result = 'failed'
            emit(receiver.inhibit('disconnected', error))
        finally:
            emit(receiver.inhibit('closed', 'Receiver ended; no active steering request'))
    summary = dict(status=result, error=error, platform=platform.platform(),
        python=sys.version.split()[0], port=name, baud=args.baud,
        stale_seconds=args.stale_seconds, valid_frames=receiver.valid_frames,
        startup_drain_seconds=args.warmup_seconds, startup_discarded_frames=startup_frames,
        startup_crc_errors=startup_decoder.crc_errors,
        rejected_frames=receiver.rejected_frames, crc_errors=receiver.decoder.crc_errors,
        elapsed_seconds=time.monotonic() - start, serial_bytes_transmitted=0,
        image_bytes_forwarded=0, motors_driven=False,
        pi_execution_verified=(platform.system() == 'Linux' and ready > 0 and result != 'failed'
            and Path('/proc/device-tree/model').exists()
            and b'Raspberry Pi' in Path('/proc/device-tree/model').read_bytes()),
        camera_angle_adjustment_verified=False, semantic_accuracy_verified=False,
        policy='Original context hysteresis + straight passage; STOP means slowdown',
        receive_interval_seconds=dict(min=min(receiver.intervals), max=max(receiver.intervals),
            mean=sum(receiver.intervals)/len(receiver.intervals)) if receiver.intervals else None,
        note='Receive intervals include USB/host buffering; not camera FPS or absolute image age.')
    (folder / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', folder)
    return 0 if result != 'failed' and ready else 1


if __name__ == '__main__':
    raise SystemExit(main())
