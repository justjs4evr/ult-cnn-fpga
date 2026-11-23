# ULT-CNN FPGA Accelerator

**LLM-assisted RTL design of a CNN inference accelerator on a low-cost FPGA (Sipeed Tang Nano 20K), deployed on a small autonomous vehicle to offload lane-scene inference from a Raspberry Pi.**

A 2,977-parameter dual-input CNN ("ULT CNN") classifies the lane scene into six driving actions plus an intersection score. Its complete integer inference datapath runs in on-chip memory of a Gowin GW2AR-18C FPGA, receives preprocessed camera frames from a XIAO ESP32S3 over UART, and returns a 36-byte result packet to the Raspberry Pi 3B+, which drives the motors through an Arduino UNO. Correctness is established in six stages, each checked against its own oracle: float model, bit-exact integer reference, RTL simulation, board replay, live camera link, and the vehicle.

> **LLM disclosure.** The RTL and verification code in this repository were generated with OpenAI Codex and verified by the author.

---

## System overview

```
 Camera A (OV5647, CSI) ──────────────────────────────► Raspberry Pi 3B+ ── YOLOv5n + control
                                                              ▲   │
 Camera B (OV3660) ─► XIAO ESP32S3 ─── UART 460,800 bps ──►  Tang Nano 20K │ USB serial
                      preprocessing    4,370-byte frame      CNN inference │ (36-byte result)
                      (contract 0x0201)                       20 MHz core   ▼
                                                             Arduino UNO + PCA9685 ─► 4 mecanum wheels
                                                             (400 ms watchdog, ultrasonic stop)
```

* **Steering input** – top 25 % of a 160×120 grayscale frame → 64×64 → binarised at 127 → 16×16 (256 B).
* **Context input** – full frame → 64×64 (4,096 B).
* **Accelerator** – one shared 16×16-bit MAC and one activation memory serve the steering path and then the context path; signed 16-bit weights/activations with per-layer Q-formats, 64-bit accumulator, 2,977 parameters stored as ten `$readmemh` files in block RAM. A frame always takes exactly 632,216 cycles (31.611 ms at 20 MHz).
* **Link** – stop-and-wait UART 8N1 frames: 16-byte header + 4,352 bytes of image data + CRC16 (4,370 B). The 36-byte result carries frame ID, timestamp, cycle count, model ID, raw class, context flags, 32-bit context score and CRC16.

Figures: [`docs/figures/`](docs/figures) (architecture, FSM, preprocessing, confusion matrix, CPU comparison, latency).

## Key results

Every number below is read from the JSON/report files linked in the last column.

| Item | Result | Evidence |
|---|---|---|
| Held-out accuracy (2,406 images, 6 classes) | **85.41 %** (macro F1 0.7772); integer model identical to float on all 2,406 | [`results/training/crossing_fix_20261011/verification_summary.json`](results/training/crossing_fix_20261011/verification_summary.json), [`results/fpga_release_20261025/quantization_report.json`](results/fpga_release_20261025/quantization_report.json) |
| Integer vs. float model (4,569 images, 5 sets) | final class 4,569 / 4,569, raw class 4,568 / 4,569, **0 saturations** | [`quantization_report.json`](results/fpga_release_20261025/quantization_report.json) |
| RTL simulation vs. integer reference | **26 / 26 frames** (13 vectors × 2), every layer tensor equal | [`results/fpga_release_20261025/rtl_verification.json`](results/fpga_release_20261025/rtl_verification.json) |
| Cycles per frame | **632,216 cycles @ 20 MHz** (31.611 ms), input-independent | `rtl_verification.json`, [`direct_link…/report.json`](results/direct_link_20261031/runs/20261031_102155_709187/report.json) |
| Board replay (PC → FPGA over UART) | **24 / 24 responses** equal (14 distinct inputs, all 6 classes); corrupted-CRC and truncated frames rejected (2 / 2) | [`results/fpga_uart_test_20261031/hardware_verification.json`](results/fpga_uart_test_20261031/hardware_verification.json) |
| Live camera link (OV3660 → XIAO → FPGA) | **5 / 5 frames**: all 4,352 preprocessed bytes and the FPGA output equal PC recomputation | [`results/direct_link_20261031/runs/20261031_102155_709187/report.json`](results/direct_link_20261031/runs/20261031_102155_709187/report.json) |
| Routed design (with camera link) | 2,970 / 20,736 logic, 962 / 15,750 registers, 44 / 46 BSRAM; 20 MHz met, Fmax 21.837 MHz, 0 setup / 0 hold violations | [`results/direct_link_20261031/cnn_direct_board_build.json`](results/direct_link_20261031/cnn_direct_board_build.json) |
| Pi CPU inference (OpenCV DNN, 1 thread, float32) | 7.363 ms per inference — faster than the FPGA core for this tiny model | [`results/cpu_comparison/pi_cpu_comparison.json`](results/cpu_comparison/pi_cpu_comparison.json) |
| **Host CPU at equal result rate** (32 results, ≈ 3.17 Hz) | one-core usage **4.541 % → 0.689 %** (−84.8 %); CPU time per result 14.190 ms → 2.160 ms | `pi_cpu_comparison.json` (`paced_CPU`, `FPGA_receiver_only`) |
| Continuous reception on the Pi (30 s) | 95 valid results, 0 CRC errors, mean interval 315.150 ms | [`results/rpi_receiver_20261031/runs/pi_live_01/summary.json`](results/rpi_receiver_20261031/runs/pi_live_01/summary.json) |
| Supervised vehicle runs (5 s / 10 s, YOLO off) | 16 / 32 valid FPGA frames, 35 / 75 motor ACKs, stop ACK verified | [`results/vehicle_control_20261031/runs/`](results/vehicle_control_20261031/runs) |
| Concurrent YOLO + FPGA (30 s, motors off) | 89 FPGA results (0 CRC errors) while YOLOv5n processed 12 frames; power-limit flag raised | [`results/pi_vision_20261031/runs/joint_after_cable/summary.json`](results/pi_vision_20261031/runs/joint_after_cable/summary.json) |

**Scope.** This is a single-design proof-of-concept. The FPGA core is slower than the Pi CPU; its value is host-CPU offload. The vehicle runs were short and supervised and establish functional integration, not driving performance. The model was trained on Webots simulator images.

## Repository layout

```
rtl/                 Verilog/SystemVerilog RTL, quant_config.vh
  weights/           ten weight/bias .hex files ($readmemh) + weights.bin
fpga/
  build.tcl          Gowin build script (top: cnn_direct_board)
  constraints/       board.cst (pins, PLL site), board.sdc (20 MHz)
  core_only_synthesis/  core-only synthesis script used during design
  reports/           archived Gowin reports (routed board design, core-only synthesis)
firmware/
  xiao/              XIAO ESP32S3 camera capture + preprocessing + UART sender (Arduino)
  uno/               Arduino UNO motor driver (PCA9685, watchdog, ultrasonic stop) + host-side unit test
host/pi/
  rpi_receiver/      result-packet decoder and receiver (CRC, staleness, policy)
  vehicle_control/   drive protocol, supervised/automatic driving, vehicle configs, unit tests
  pi_vision/         YOLOv5n worker + concurrent lane/vision run
  cpu_comparison/    ONNX export, fixtures and the Pi CPU-vs-FPGA benchmark
  CAR_START.py       launcher (check / ports / preview / fixed / drive / stop)
model/
  model.py, inference.py, train_*.py, finetune_*.py   PyTorch model, training and fine-tuning
  fpga_export/       fixed-point conversion (export_fpga.py) and integer reference (predict_integer.py)
  user_runs/         trained checkpoints (final: crossing_fix_20261011/best_model.pt, 18 KB)
  webots/            Webots worlds, track images, controllers and dataset tooling
verification/
  tb/                core_tb.sv, packet_tb.sv, stream_tb.sv
  vectors/           13 test vectors with every intermediate layer tensor
  logs/              archived simulation logs
  verify_rtl.py      runs all vectors through RTL simulation (Icarus Verilog)
results/             JSON evidence behind the numbers above
docs/                figures, hardware pin reference, provenance notes
```

See [`docs/PROVENANCE.md`](docs/PROVENANCE.md) for where every file came from and which path constants were adapted for this layout.

## Hardware

| Part | Role |
|---|---|
| Sipeed **Tang Nano 20K** (Gowin GW2AR-LV18QN88C8/I7, 20,736 LUT4, 46 BSRAM) | CNN inference accelerator, 27 MHz board clock → 20 MHz PLL |
| Seeed **XIAO ESP32S3** + **OV3660** camera | lane camera (B), preprocessing, UART frame sender |
| **Raspberry Pi 3B+** + OV5647 CSI camera | host: YOLOv5n object detection (camera A), result receiver, driving decisions |
| **Arduino UNO** + **PCA9685** PWM driver | motor control, 400 ms watchdog, ultrasonic stop at 5 cm (TRIG D11 / ECHO D12) |
| Four-wheel mecanum chassis (Roboduino V2 class) | vehicle |

Wiring (from [`direct_stage_status.json`](results/direct_link_20261031/direct_stage_status.json) and [`board.cst`](fpga/constraints/board.cst)):

| From | To |
|---|---|
| XIAO D6 / GPIO43 (TX) | Tang pin 73 (header J6-1, `xiao_rx`) |
| XIAO D7 / GPIO44 (RX) | Tang pin 74 (header J6-2, `xiao_tx`) |
| XIAO GND | Tang GND |
| Tang onboard USB (BL616 debugger UART, pin 69 `uart_tx`) | Raspberry Pi USB |
| Arduino UNO USB | Raspberry Pi USB |

Pin label reference: [`docs/hardware/Tang_pin_label.png`](docs/hardware/Tang_pin_label.png). Schematic: [Sipeed Tang Nano 20K wiki](https://wiki.sipeed.com/hardware/en/tang/tang-nano-20k/nano-20k.html).

## Build and run

### 1. RTL simulation (reproduces the 26/26 result)

Requires [Icarus Verilog](https://steveicarus.github.io/iverilog/) (`iverilog`, `vvp` on `PATH`, or set `IVERILOG_BIN`).

```bash
python verification/verify_rtl.py
# -> PASS lines for 13 vectors x 2 frames, cycles=632216; report in verification/out/rtl_verification.json
```

### 2. FPGA bitstream

Requires Gowin EDA (`gw_sh`).

```bash
gw_sh fpga/build.tcl          # outputs to rtl/impl/pnr/cnn_direct_board.fs
openFPGALoader -b tangnano20k -f rtl/impl/pnr/cnn_direct_board.fs   # write to flash
```

The original bitstream (SHA-256 `65150b3c…`, listed in `cnn_direct_board_build.json`) is not committed; it can be attached to a GitHub release.

### 3. XIAO ESP32S3 firmware

```bash
arduino-cli compile --fqbn "esp32:esp32:XIAO_ESP32S3:PSRAM=opi,USBMode=hwcdc,CDCOnBoot=default,PartitionScheme=default_8MB" firmware/xiao/ULT_Direct_Camera
arduino-cli upload  --fqbn "esp32:esp32:XIAO_ESP32S3:PSRAM=opi,USBMode=hwcdc,CDCOnBoot=default,PartitionScheme=default_8MB" -p <PORT> firmware/xiao/ULT_Direct_Camera
```

### 4. Arduino UNO firmware

Requires the *Adafruit PWM Servo Driver* library.

```bash
arduino-cli compile --fqbn arduino:avr:uno firmware/uno/ULT_Vehicle_Control
arduino-cli upload  --fqbn arduino:avr:uno -p <PORT> firmware/uno/ULT_Vehicle_Control
```

### 5. Raspberry Pi host

```bash
cd host/pi
pip install -r requirements_control.txt
python3 CAR_START.py check              # configuration and file checks
python3 CAR_START.py ports              # list FPGA / UNO serial ports
python3 CAR_START.py preview            # receive FPGA results without driving motors
python3 rpi_receiver/receiver.py --seconds 30 --output run.json   # stand-alone receiver
python3 -m unittest discover vehicle_control   # host-side unit tests
```

Driving modes (`fixed`, `drive`) require `--clear-floor --motor-power-on` and should only be run under supervision. Speeds are set by the six PWM values in `DRIVE_SETTINGS.json`.

### 6. Model training and fixed-point export

```bash
pip install -r model/requirements.txt
python model/train_classified_images.py --help
python model/fpga_export/export_fpga.py      # Q-format conversion, weights, vectors, integer-vs-float report
```

`export_fpga.py` writes its outputs (weights, `quant_config.vh`, vectors, report) into `model/fpga_export/`; the committed copies live in `rtl/`, `verification/vectors/` and `results/`.

## Data

The training dataset (23,948 Webots images in six classes, hand-labelled, split by input hash into 19,169 / 2,373 / 2,406) is not included because of its size. **Data is available on request.**

## Citation

The accompanying paper is in preparation:

```bibtex
@unpublished{choi2027ultcnn,
  author = {Doyoun Choi},
  title  = {From {LLM}-Generated {RTL} to a Moving Vehicle: Staged Verification of a Low-Cost {FPGA} {CNN} Accelerator for Host-{CPU} Offload},
  note   = {Manuscript in preparation},
  year   = {2027}
}
```

## Acknowledgment

The RTL of the accelerator was generated by OpenAI Codex under the author's direction. Third-party components: [pyserial](https://github.com/pyserial/pyserial) (BSD-3-Clause, installed via pip), Adafruit PWM Servo Driver library, ESP32 camera driver, YOLOv5n (Ultralytics), Webots.
