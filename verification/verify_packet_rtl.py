import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
def main():
    tools=HERE/'tools/iverilog/bin';build=HERE/'simulation';build.mkdir(exist_ok=True)
    commands=[
        [str(tools/'iverilog.exe'),'-g2012','-s','packet_tb','-o',str(build/'packet.vvp'),str(HERE/'rtl/result_packet_tx.sv'),str(HERE/'rtl/packet_tb.sv')],
        [str(tools/'vvp.exe'),str(build/'packet.vvp')]]
    logs=[]
    for command in commands:
        result=subprocess.run(command,cwd=HERE,text=True,capture_output=True);logs.append(result.stdout+result.stderr)
        result.check_returncode()
    if 'PASS packet' not in logs[-1]:raise RuntimeError('Missing PASS')
    (build/'packet.log').write_text('\n'.join(logs),encoding='utf-8')
    (HERE/'packet_rtl_verification.json').write_text(json.dumps(dict(status='passed',bytes=36,crc=True,ready_stalls=True),indent=2),encoding='utf-8')
    print(logs[-1])
if __name__=='__main__':main()
