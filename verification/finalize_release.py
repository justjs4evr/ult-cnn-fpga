"""Gate and package measured export artifacts; never program a board."""
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

HERE=Path(__file__).resolve().parent


def read(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
def dump(name,data):(HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest=read('manifest.json');quant=read('quantization_report.json');rtl=read('rtl_verification.json')
    synth=read('synthesis_report.json');protocol=read('protocol_verification.json');packet=read('packet_rtl_verification.json')
    source_hash=sha(HERE/'rtl/ult_context_core.sv')
    assert rtl['status']=='passed' and rtl['vectors']==13 and rtl['frames']==26
    assert source_hash==rtl['rtl_sha256']==synth['rtl_sha256']
    assert sha(HERE/'rtl/quant_config.vh')==rtl['quant_config_sha256']
    assert protocol['status']==packet['status']=='passed'
    assert sha(Path(manifest['checkpoint']))==manifest['checkpoint_sha256']
    assert all(g['images']==g['final_class_matches']==g['activate_matches']==g['keep_matches'] for g in quant['groups'].values())
    assert not any(quant['saturation'].values())
    payload=bytearray()
    for tensor in manifest['tensors']:
        path=HERE/tensor['file'];values=[int(line,16) for line in path.read_text(encoding='ascii').split()]
        assert len(values)==tensor['count']
        tensor['sha256']=sha(path);tensor['binary_offset_bytes']=len(payload)
        for value in values:payload.extend(value.to_bytes(2,'little'))
    assert len(payload)==2977*2
    (HERE/'weights.bin').write_bytes(payload)
    manifest['packed_weights']=dict(file='weights.bin',bytes=len(payload),endianness='little',signed_bits=16,
        sha256=sha(HERE/'weights.bin'),note='Packed parameter values only; not a board configuration or bitstream.')
    manifest['status']='verified_core_export_not_board_ready'
    dump('manifest.json',manifest)
    shutil.copy2(manifest['checkpoint'],HERE/'model_source.pt')
    groups=quant['groups'];cycles=rtl['results'][0]['cycles']
    summary=dict(status='verified_core_export_not_board_ready',source_checkpoint=manifest['checkpoint'],
        source_checkpoint_sha256=manifest['checkpoint_sha256'],quantization=groups,
        saturated_values=0,rtl_vectors=13,rtl_frames=26,inference_cycles_excluding_input=cycles,
        core_time_at_proposed_20mhz_ms=cycles/20000,hardware=synth,
        packet_bytes=36,protocol_version=2,board_programmed=False,
        remaining=['Camera B initialization, capture, CDC and matching dual-view preprocessing.',
                   'UART physical transmitter, core/packet integration and actual RPi serial test.',
                   '20MHz PLL/clock selection, pins, full-design routing/timing and board playback.',
                   'Native integer-model closed-loop and real-camera validation.'])
    dump('release_summary.json',summary)
    guide=f'''ULT-CNN FPGA 변환 패키지 (2026-10-25)

1. 현재 상태
검증한 주행 모델의 전체 CNN과 교차로 인식부를 16비트 가중치로 변환했습니다.
파라미터 2,977개, weights.bin 5,954바이트이며, 바로 보드에 쓰는 비트스트림은 아닙니다.
가중치 ROM 10개 HEX, 정수 참조 Python, SystemVerilog 코어, 테스트벤치,
36바이트 결과 패킷 RTL, Raspberry Pi 수신·교차로 유지 코드를 포함합니다.

2. 핵심 파일
weights/*.hex: 각 층의 가중치·bias. 한 줄에 signed int16 2의 보수 4자리 HEX.
weights.bin: manifest 순서대로 이어 붙인 little-endian signed int16 파라미터.
manifest.json: 원본 모델 SHA256, 형상·순서·고정 소수점 스케일·임계값·파일 해시.
model_source.pt: 현재 주행 중인 원본 모델의 보관본.
rtl/ult_context_core.sv: 기본 CNN과 전체 화면 교차로 인식부를 모두 연산.
rtl/quant_config.vh: 층별 이동 비트와 교차로 활성/유지 정수 임계값.
rtl/result_packet_tx.sv: 클래스·교차로 점수·모델 ID를 CRC와 함께 바이트로 출력.
rpi_receiver.py: 표준 Python 수신기. 1.5초 직진 유지·정상 조향 복귀; GPIO 구동 없음.
quantization_report.json / rtl_verification.json / synthesis_report.json: 검증 결과.

3. 검증 결과
저장 영상 {quant['input_count']:,}장의 최종 클래스, 활성 임계값, 유지 임계값 판단이 원본과 모두 일치.
기존 사진 2,406장 정확도: 원본/정수 모두 85.41%. 이것은 분류 정확도 100%를 뜻하지 않습니다.
기본 CNN 단독 출력은 전체 중 1장 차이가 있었지만 교차로 판단을 반영한 최종 클래스는 일치했습니다.
출력 포화 0회. 실제 사진 8개+합성 진단 5개, 총 13개 벡터를 각각 두 번 연속 재생.
26프레임의 모든 합성곱·풀링·FC 출력, 교차로 점수와 클래스가 정수 참조와 비트 단위 일치.
실제 카메라 입력과 폐루프 주행을 정수 모델로 새로 측정한 결과는 아닙니다.

4. Tang Nano 20K 코어 합성
장치: {synth['device']}
Logic: {synth['resources']['Logic']}
Register: {synth['resources']['Register']}
BSRAM: {synth['resources']['BSRAM']}
합성 단계 예상 Fmax: {synth['synthesis_fmax_estimate_mhz']:.3f}MHz.
기본 100MHz 제약을 만족하지 않으며 보드의 27MHz를 그대로 사용한다고 가정하면 안 됩니다.
20MHz 코어 클럭을 제안합니다. 코어 {cycles:,}사이클은 20MHz 가정 시 {cycles/20000:.2f}ms입니다.
입력 전송·카메라·전처리·UART·서보 지연은 별도이며 이 수치는 실측 FPS가 아닙니다.
BSRAM이 87%여서 카메라 전처리/FIFO의 추가 사용량도 확인해야 합니다.
배치·배선 및 실제 클럭/핀 제약 검증은 하지 않았습니다.

5. FPGA 입력 계약 (같은 카메라 B 영상의 두 버전)
input_start 펄스 후 pixel_valid/pixel_ready로 16x16 256바이트,
이어 전체 화면 64x64 4,096바이트를 행 우선으로 입력합니다. 총 4,352바이트.
기본 입력: RGB->PIL L, 상단 floor(H*0.25) 제외, BOX 64x64,
127 이상을 255/나머지 0 이진화, BOX 16x16. 평균 축소여서 중간 회색값이 존재합니다.
교차로 입력: 원본 전체 화면 PIL L, BOX 64x64, 이진화·위쪽 자르기 없음.
카메라 A는 RPi YOLO용이며 이 패키지의 두 입력과 별개입니다.
FPGA 카메라 RGB565->회색/BOX 평균/반올림이 이 계약과 같아지는지 별도 픽셀 대조가 필요합니다.
현재는 이미 전처리된 바이트를 받는 코어입니다. 카메라에 연결한 top 프로젝트가 아닙니다.

6. 고정 소수점
입력: (byte*2048+127)//255, Q11.
활성값: conv1 Q11, conv2 Q10, FC1 Q8, FC2 Q9, context logit signed int32 Q11.
가중치: conv1/conv2/FC2/context Q14, FC1 Q13. bias는 해당 출력과 같은 스케일.
MAC signed int64; 출력 = ties-away-from-zero(MAC >> SHIFT)+bias, ReLU 후 int16 포화.
context 출력만 int32 포화. 스케일은 manifest와 quant_config를 함께 사용합니다.
Conv는 커널 반전 없는 O/I/Y/X, FC는 O/I, feature flatten은 C/Y/X.
최댓값 동점이면 작은 클래스 번호. 클래스 순서: straight,left,right,sharp_left,sharp_right,stop.
교차로 활성 context_logit_q >= {manifest['thresholds']['activate_q']}; 유지 >= {manifest['thresholds']['keep_q']}.
sigmoid/softmax를 RTL에서 계산하지 않습니다. 확률처럼 보이는 값을 만들어 보내지 않습니다.

7. FPGA -> Raspberry Pi 통신 (새 버전 2)
36바이트 little-endian. 기존 문서의 v1 32바이트와 구분해야 합니다.
0..1 AA55 / 2 version=2 / 3 type=1
4..7 frame_id uint32 / 8..11 capture_tick uint32 (촬영 ms, 32bit wrap)
12..15 inference_us uint32 / 16..19 model_id uint32
20 기본 CNN raw_class uint8 / 21 flags uint8 / 22..23 preprocess_id uint16=0x0201
24..27 capture_to_tx_us uint32 / 28..31 context_logit_q signed int32 Q11
32..33 reserved=0 / 34..35 CRC16-CCITT-FALSE (앞 34바이트, CRC도 little-endian).
model_id는 checkpoint SHA256의 첫 4바이트를 little-endian 해석한 값입니다.
flags bit0 valid, bit1 camera_error, bit2 overflow, bit3 saturation,
bit4 context_activate, bit5 context_keep, bit6..7 예약0.
raw_class 포트와 context_logit_q를 패킷에 연결합니다.
class_result는 단일 프레임 최종 분류 확인용이며 raw_class 대신 보내지 않습니다.
packet_start에는 코어 result_valid와 동기화된 메타데이터를 연결합니다.
byte_ready는 별도 UART 송신기의 수락 신호입니다. UART 전기 신호 TX 구현은 별도입니다.

8. Raspberry Pi 제어 역할
활성 점수에 도달하면 직진을 유지하고, keep 임계값 이상이면 유지 시간을 갱신합니다.
keep 미만이 1.5초 지속되면 기본 CNN 조향으로 복귀합니다. 이 시간은 정지 대기가 아닙니다.
현재 시뮬레이션 정책대로 raw stop도 straight로 변환하고 slowdown_event를 별도로 알립니다.
실차는 slowdown_event를 속도 제어와 결합하고 YOLO 명령·미션 경로와 함께 최종 조향을 결정합니다.
수신기는 모델 ID/전처리 ID/CRC/프레임 순서/점수 플래그를 검사합니다.
수신 공백 0.3초면 상태를 지우며, 이 기본 시간은 실측 지연을 보고 조정합니다.
GPIO/모터 배선·PWM 값은 이 패키지가 설정하지 않습니다.
RPi로 rpi_receiver.py와 manifest.json을 복사하면 Torch 없이 사용할 수 있습니다.
실제 UART 사용에는 pyserial 설치 후: python3 rpi_receiver.py --port /dev/serial0
포트 이름·baud 115200과 3.3V/GND 연결은 실제 보드 구성에서 확인합니다.

9. PC 실행
기존 환경 C:\\AICAR\\ULT_CNN\\.venv를 사용합니다.
VERIFY_FPGA.cmd: 프로토콜 + 패킷 RTL + 13벡터 CNN RTL 검증. 수 분 소요.
EXPORT_MODEL.cmd: 고정된 원본 모델에서 정수 가중치와 4,569장 비교를 다시 생성.
python predict_integer.py <사진 경로>: HEX를 실제로 읽어서 새 사진 정수 예측.
python run_synthesis.py: 설치된 C:\\Gowin\\IDE 도구로 코어만 합성.
내보내기를 다시 하면 검증/합성을 다시 하고 python finalize_release.py로 요약·해시를 갱신합니다.
도구: tools/iverilog 안의 프로젝트 전용 Icarus Verilog. 전역 PATH는 필요 없습니다.
배포 ZIP은 도구/학습 데이터/중간 합성 파일을 제외한 소스·가중치·보고서입니다.
ZIP을 다른 PC에 풀었다면 tools/install_iverilog.ps1로 프로젝트 전용 시뮬레이터를 설치할 수 있습니다.

10. 다음 보드 작업
카메라 B 초기화/DVP 수신/CDC/동일 전처리와 코어 입력을 연결합니다.
결과 패킷과 실제 UART를 연결하고, 20MHz 클럭·핀 제약으로 전체 설계를 합성·배치·배선합니다.
비트스트림을 만들기 전에 저장 HEX 입력을 보드에서 재생해 PC 정수 결과와 비교합니다.
그 다음 실제 카메라 픽셀·RPi 수신·폐루프 주행을 검증합니다.
원본 모델과 Webots 선택 설정은 변경하지 않았습니다.
'''
    (HERE/'사용안내.txt').write_text(guide,encoding='utf-8')
    table=''.join(f'<tr><td>{html.escape(name)}</td><td>{g["images"]:,}</td><td>{g["final_class_matches"]:,}</td><td>{g["activate_matches"]:,} / {g["keep_matches"]:,}</td></tr>' for name,g in groups.items())
    report=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>ULT-CNN FPGA 변환 결과</title>
<style>body{{max-width:1040px;margin:32px auto;padding:0 24px;font-family:Malgun Gothic,Segoe UI,sans-serif;line-height:1.7;background:#f3f6fa;color:#17324a}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}code{{overflow-wrap:anywhere}}a{{color:#1262a0}}</style>
<h1>ULT-CNN FPGA 변환 결과</h1><p>2026-10-25 · 2,977개 파라미터 · Tang Nano 20K 코어 검증</p>
<section><h2>완료한 범위</h2><p>검증된 주행 모델을 16비트 가중치로 내보냈습니다. 기존 CNN과 64×64 교차로 인식부를 모두 연산하는 SystemVerilog 코어, 결과 패킷, RPi 수신·유지 제어 코드를 만들었습니다. 원본 모델과 Webots 선택은 그대로입니다.</p><p><a href="사용안내.txt">전체 사용·연산·통신 규격</a> · <a href="manifest.json">모델/가중치 manifest</a> · <a href="weights.bin">가중치 바이너리</a> · <a href="rtl/ult_context_core.sv">CNN RTL</a> · <a href="rpi_receiver.py">RPi 수신기</a></p><p>weights.bin은 5,954바이트의 파라미터 데이터입니다. 보드에 바로 쓰는 비트스트림이 아닙니다.</p></section>
<section><h2>정수 변환 비교</h2><table><tr><th>입력 묶음</th><th>사진 수</th><th>최종 클래스 일치</th><th>활성 / 유지 판단 일치</th></tr>{table}</table><p>총 4,569장의 저장 영상에서 최종 클래스와 두 임계값 판단이 모두 일치했고 포화는 0회였습니다. 기본 CNN 단독 출력은 1장에서 차이가 났지만 교차로 인식 후 최종 출력은 일치했습니다. 기존 사진 2,406장의 정답 정확도는 원본·정수 모두 85.41%입니다. 원본과의 일치율 100%가 정답 정확도 100%를 뜻하지는 않습니다.</p><p>저장 영상 재생 결과입니다. 정수 모델로 실제 카메라 폐루프 주행을 새로 검증한 결과는 아닙니다.</p><p><a href="quantization_report.json">정수 비교 보고서</a> · <a href="predictions.csv">사진별 비교</a></p></section>
<section><h2>RTL 검증</h2><p>실제 영상 8개와 0·255·체커·난수·테두리 진단 5개를 각각 두 번 연속 입력했습니다. 총 26프레임에서 모든 층 출력, 교차로 점수, 클래스가 정수 참조와 비트 단위로 일치했습니다. 입력 valid 공백과 RAM 재사용도 검사했습니다.</p><p>36바이트 결과 패킷은 Python 기준 바이트·CRC와 일치했고, 수신기는 잡음·분할 수신·CRC 오류·프레임 순서·시간 만료를 검사했습니다.</p><p><a href="rtl_verification.json">CNN RTL 결과</a> · <a href="protocol_verification.json">수신 정책 결과</a> · <a href="packet_rtl_verification.json">패킷 RTL 결과</a></p></section>
<section><h2>보드 자원과 남은 작업</h2><p>코어만 GOWIN 합성에 성공했습니다. Logic {html.escape(synth['resources']['Logic'])}, Register {html.escape(synth['resources']['Register'])}, BSRAM {html.escape(synth['resources']['BSRAM'])}입니다. 메모리 87% 사용이므로 카메라·FIFO 추가분을 확인해야 합니다.</p><p>합성 예상 Fmax는 {synth['synthesis_fmax_estimate_mhz']:.3f}MHz입니다. 기본 100MHz 제약은 만족하지 않았으며, 보드의 27MHz를 그대로 사용할 수 있다고 가정하지 않습니다. 20MHz 코어 클럭을 제안하며, 이 경우 연산만 약 {cycles/20000:.2f}ms입니다. 배치·배선 후 타이밍이나 실측 FPS는 아직 확인하지 않았습니다.</p><p>카메라 입력·전처리·CDC, 실제 UART, 클럭·핀 연결, 전체 설계 배치·배선과 보드 시험을 거쳐 비트스트림을 만들어야 합니다. 현재 폴더에는 .fs 비트스트림이 없습니다.</p><p><a href="synthesis_report.json">합성 요약</a> · <a href="impl/gwsynthesis/ult_context_core_syn.rpt.html">GOWIN 원본 보고서</a></p></section>
<section><h2>차량 역할 분담</h2><p>카메라 A → RPi YOLO. 카메라 B → FPGA에서 기본 조향 CNN + 교차로 인식 CNN → 기본 클래스와 교차로 점수를 RPi로 전송합니다. RPi가 1.5초 유지·감속 이벤트·YOLO·차량별 조향값을 결합합니다. 카메라 B의 같은 영상을 16×16과 64×64로 처리하며 물리 카메라를 추가하지 않습니다.</p><p>통신은 교차로 점수를 추가한 v2 36바이트 규약입니다. 기존 문서의 v1 32바이트 수신기와 구분해야 합니다. RPi 코드는 출력 클래스와 slowdown_event까지만 제공하며 GPIO를 구동하지 않습니다.</p><p><code>{html.escape(str(HERE))}</code></p></section></html>'''
    (HERE/'FPGA_변환_검증결과.html').write_text(report,encoding='utf-8')
    # Small transfer bundle: preserve exact sources/weights/vectors/reports, omit tools and build intermediates.
    archive=HERE.parent/'ULT_CNN_FPGA_20261025.zip'
    files=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.sv','.vh','.json','.txt','.html','.cmd','.bin','.hex','.pt','.tcl','.csv')]
    files.extend([HERE/'impl/gwsynthesis/ult_context_core_syn.rpt.html',HERE/'impl/gwsynthesis/ult_context_core.log'])
    files.append(HERE/'tools/install_iverilog.ps1')
    files.extend((HERE/'simulation_final').glob('*.log'))
    files.append(HERE/'simulation/packet.log')
    for directory in ('weights','rtl','vectors'):
        files.extend(p for p in (HERE/directory).rglob('*') if p.is_file())
    dump('file_hashes.json',{str(p.relative_to(HERE)):sha(p) for p in files if p.name!='file_hashes.json'})
    if HERE/'file_hashes.json' not in files:files.append(HERE/'file_hashes.json')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,'fpga_release_20261025/'+str(p.relative_to(HERE)))
    dump('../latest_fpga_export.json',dict(directory=str(HERE),archive=str(archive),status=summary['status'],checkpoint_sha256=manifest['checkpoint_sha256']))
    print(json.dumps(dict(directory=str(HERE),archive=str(archive),status=summary['status']),ensure_ascii=False))


if __name__=='__main__':main()
