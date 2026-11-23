"""Run all exported vectors against every RTL layer and both outputs."""
import hashlib
import json
from pathlib import Path
import os
import re
import shutil
import subprocess

# Repository layout (paths adapted from the original fpga_release_20261025 package):
#   rtl/            RTL + quant_config.vh + weights/*.hex  ($readmemh paths are relative to rtl/)
#   verification/   this script, tb/, vectors/
HERE=Path(__file__).resolve().parent
RTL=HERE.parent/'rtl'
TB=HERE/'tb'


def tool(name):
    # Icarus Verilog from IVERILOG_BIN, else from PATH.
    folder=os.environ.get('IVERILOG_BIN')
    path=shutil.which(name,path=folder) if folder else shutil.which(name)
    if not path:raise SystemExit(f'{name} not found: install Icarus Verilog or set IVERILOG_BIN')
    return path


def main():
    source_hash=hashlib.sha256((RTL/'ult_context_core.sv').read_bytes()).hexdigest()
    config_hash=hashlib.sha256((RTL/'quant_config.vh').read_bytes()).hexdigest()
    build=HERE/'out/simulation_final';build.mkdir(parents=True,exist_ok=True)
    command=[tool('iverilog'),'-g2012','-Wall','-I',str(RTL),'-s','core_tb',
        '-o',str(build/'core.vvp'),str(RTL/'ult_context_core.sv'),str(TB/'core_tb.sv')]
    compile_result=subprocess.run(command,cwd=RTL,text=True,capture_output=True)
    (build/'compile.log').write_text(compile_result.stdout+compile_result.stderr,encoding='utf-8')
    compile_result.check_returncode()
    vectors=json.loads((HERE/'vectors/manifest.json').read_text(encoding='utf-8'))
    results=[]
    for item in vectors:
        vector=(HERE/item['path'].replace('\\','/')).as_posix()
        result=subprocess.run([tool('vvp'),str(build/'core.vvp'),'+VECTOR='+vector],
            cwd=RTL,text=True,capture_output=True,timeout=120)
        (build/(item['name']+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        result.check_returncode()
        lines=[line for line in result.stdout.splitlines() if line.startswith('PASS ')]
        if len(lines)!=2:raise RuntimeError('Missing two-frame PASS')
        results.append(dict(vector=item['name'],frames=2,cycles=int(re.search(r'cycles=(\d+)',lines[-1]).group(1)),
                            passed=True))
        print(lines[-1],flush=True)
    if source_hash!=hashlib.sha256((RTL/'ult_context_core.sv').read_bytes()).hexdigest():
        raise RuntimeError('RTL changed during simulation; rerun verification')
    if config_hash!=hashlib.sha256((RTL/'quant_config.vh').read_bytes()).hexdigest():
        raise RuntimeError('Quantization config changed during simulation')
    report=dict(status='passed',vectors=len(results),frames=sum(r['frames'] for r in results),
        checked='Every conv/pool/dense tensor, context logit, both classes, threshold flags; repeated frames with input-valid stalls.',
        rtl_sha256=source_hash,quant_config_sha256=config_hash,results=results,
        limitations='RTL simulation only; not synthesis, timing closure or physical camera/board validation.')
    # Written to verification/out/ so the archived evidence in results/ is never overwritten.
    (HERE/'out/rtl_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
