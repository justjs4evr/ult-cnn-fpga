"""Synthesize core only and preserve exact RTL identity and tool output."""
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent


def main():
    source_hash=hashlib.sha256((HERE/'rtl/ult_context_core.sv').read_bytes()).hexdigest()
    result=subprocess.run([r'C:\Gowin\IDE\bin\gw_sh.exe',str(HERE/'gowin_synthesis.tcl')],
        cwd=HERE,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (HERE/'synthesis_console.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    result.check_returncode()
    path=HERE/'impl/gwsynthesis/ult_context_core_syn.rpt.html'
    document=path.read_text(encoding='utf-8')
    section=document.split('name="utilization"',1)[1].split('</table>',1)[0]
    resources={}
    for name in ('Logic','Register','BSRAM'):
        match=re.search(r'<td class="label">'+name+r'</td>\s*<td>(.*?)</td>',section,re.S)
        if not match:raise RuntimeError('Missing synthesis resource: '+name)
        resources[name]=html.unescape(re.sub('<[^>]+>','',match.group(1)))
    timing=document.split('name="performance"',1)[1].split('</table>',1)[0]
    cells=re.findall(r'<td>(.*?)</td>',timing,re.S)
    fmax=float(re.search(r'[0-9.]+',cells[3]).group())
    if source_hash!=hashlib.sha256((HERE/'rtl/ult_context_core.sv').read_bytes()).hexdigest():
        raise RuntimeError('Source changed during synthesis')
    report=dict(status='core_synthesis_completed',device='GW2AR-LV18QN88C8/I7',
        rtl_sha256=source_hash,resources=resources,synthesis_fmax_estimate_mhz=fmax,
        tool_default_constraint_mhz=100,timing_closed=False,place_and_route_completed=False,
        bitstream_generated=False,report=str(path),
        limitation='Pre-layout core estimate. Default 100MHz constraint is not met. A 20MHz core clock is a proposal requiring PLL and full-design routed timing validation.')
    (HERE/'synthesis_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
