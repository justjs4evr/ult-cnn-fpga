# Provenance

This repository was assembled on 2026-11-23 by **copying** files from the author's project archives (nothing was moved or deleted at the source). RTL, weights and firmware were byte-identical (SHA-256) across all archived copies.

## Sources

| Source archive | Used for |
|---|---|
| `handoff_20261101_CNN_vehicle/AICAR_CNN_Vehicle` (final hand-off, 2026-11-01) | `rtl/`, `rtl/weights/`, `fpga/constraints/`, `fpga/build.tcl`, `fpga/reports/direct_board/`, `firmware/`, `host/pi/` (newest versions of `vehicle_control`, with `--fpga-port`), `model/*.py`, final checkpoint, vehicle evidence |
| `research_paper_20261101` (paper evidence bundle) | `results/` (most JSON), `host/pi/pi_vision/`, `host/pi/cpu_comparison/`, training scripts, upstream checkpoints, `docs/figures/` |
| `handoff_20261025/AICAR` (FPGA release hand-off, 2026-10-25) | `verification/` (testbenches, 13 vectors, simulation logs, verify scripts), `results/fpga_release_20261025/synthesis_report.json` and related reports, `fpga/core_only_synthesis/`, `fpga/reports/core_only/`, Webots worlds/controllers/tooling |

The original directory names (e.g. `fpga_release_20261025`, `direct_link_20261031`) are kept under `results/` so that evidence files match the paths referenced inside them.

## Path adaptations

Only path constants were changed so that the code runs from this layout; no logic was changed.

| File | Change |
|---|---|
| `host/pi/CAR_START.py` | `ULT_CNN/vehicle_control_20261031` → `vehicle_control` |
| `host/pi/CAMERA_USB.py` | `ULT_CNN/rpi_receiver_20261031/vendor` → `rpi_receiver/vendor` (pyserial now comes from pip) |
| `host/pi/vehicle_control/*.py`, `host/pi/pi_vision/run_joint.py` | `rpi_receiver_20261031` → `rpi_receiver` |
| `host/pi/cpu_comparison/benchmark_cnn_on_pi.py` | receiver path `~/AICAR/ULT_CNN/rpi_receiver_20261031` → repository-relative `host/pi/rpi_receiver` |
| `host/pi/cpu_comparison/prepare_cnn_cpu_comparison.py` | `C:\AICAR\…` paths → repository-relative (`model/`, `verification/vectors/`, output next to the script) |
| `host/pi/requirements_control.txt` | comment updated (vendored pyserial not included) |
| `model/evaluate_images.py` | default dataset `C:\AICAR\main_classified_images` → `data/main_classified_images` |
| `verification/verify_rtl.py` | uses `../rtl` and `tb/`, finds Icarus Verilog on `PATH`/`IVERILOG_BIN` instead of a bundled copy, writes to `verification/out/` |
| `fpga/build.tcl` | runs from `rtl/` so `$readmemh("weights/…")` resolves; constraints from `fpga/constraints/` |

After these changes: `verification/verify_rtl.py` reproduced **26/26 PASS, 632,216 cycles** with Icarus Verilog, and the host unit tests (`python -m unittest discover vehicle_control`) passed **26 tests**. The adapted `fpga/build.tcl` has **not** been re-run (Gowin EDA was not available at assembly time).

## Sanitisation

* `results/rpi_receiver_20261031/pi_execution.json`: the Raspberry Pi LAN address and Linux user name were replaced with `<PI_HOST>` and `<PI_USER>`.
* No passwords, API keys, Wi-Fi credentials or e-mail addresses were found in the copied code. `token` in the UNO protocol is a per-session random number, not a secret.
* Absolute paths of the form `C:\AICAR\…` inside archived logs/JSON were left unchanged as part of the evidence; they contain no personal information.

## Not included

| Item | Reason |
|---|---|
| Training/evaluation images (23,948 labelled Webots images; raw captures; `AICAR_*데이터*.zip`, ~2.3 GB) | size — available on request |
| `cnn_direct_board.fs` bitstream (7.3 MB) | build artefact — to be attached to a GitHub release |
| ESP32 build outputs (`.elf`, `.map`, `.merged.bin`), UNO `.hex` | build artefacts |
| Bundled Icarus Verilog binaries, Python `.venv`, vendored pyserial sources | third-party tools; install separately |
| `predictions.csv`, dataset split manifests (1–3 MB CSV) | derived from the dataset |
| Webots auto-collection job/status logs and episode metadata | runtime logs |
| Paper PDFs and LaTeX sources | withheld for now |

## Known gaps

* `verification/tb/stream_tb.sv` reads `test_data/*.hex` files that were not present in any archive; that testbench is included for reference only.
* The following items referenced in the author's notes were not found in the archives: `fpga_board_test_20261031/` (LED/PLL build log, `physical_observations.json`), `asic_scenarios.json`, `worked_mac_example.json`, and direct-link run `20261031_102534_556656`.
* The direct-link run included here (`20261031_102155_709187`) has a 5-frame mean round trip of 128.692 ms; `direct_stage_status.json` records a separate 5-frame measurement with mean 128.723 ms (min 128.309, max 129.119 ms).
