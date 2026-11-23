# Core-only synthesis; no pin assignment, routing, bitstream or programming.
cd [file dirname [info script]]
set_device -name GW2AR-18C GW2AR-LV18QN88C8/I7
add_file -type verilog "rtl/ult_context_core.sv"
set_option -top_module ult_context_core
set_option -verilog_std sysv2017
set_option -include_path {rtl}
set_option -output_base_name ult_context_core
run syn
