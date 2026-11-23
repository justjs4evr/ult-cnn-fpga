"""XIAO persisted start/stop or info. No motor commands are implemented."""
import argparse
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'rpi_receiver/vendor'))
import serial
from serial.tools import list_ports

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['info','start','stop'])
    ap.add_argument('--port')
    args = ap.parse_args()
    candidates = [p.device for p in list_ports.comports() if p.vid in (0x303a,0x2886)]
    name = args.port or (candidates[0] if len(candidates)==1 else None)
    if not name:
        ap.error('Specify the verified XIAO USB port using --port')
    with serial.Serial(port=None, baudrate=115200, timeout=.05, write_timeout=2) as port:
        port.dtr=False; port.rts=False; port.port=name; port.open(); port.dtr=True
        time.sleep(.2)
        port.reset_input_buffer()
        port.write({'info':b'I','start':b'G','stop':b'S'}[args.action]); port.flush()
        deadline = time.monotonic()+3
        output = bytearray()
        while time.monotonic()<deadline:
            output.extend(port.read(max(1,min(4096,port.in_waiting))))
            if b'ULTCAM_INFO ' in output or b'ULT_DIRECT auto=' in output:
                print(output.decode('utf-8', errors='replace').strip())
                return 0
        print(output.decode('utf-8', errors='replace'))
        raise RuntimeError('No firmware response. Check XIAO USB data cable and port.')

if __name__ == '__main__':
    raise SystemExit(main())
