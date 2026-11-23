# Gowin build script for the Tang Nano 20K CNN accelerator (top: cnn_direct_board).
# Usage (Gowin EDA installed):  gw_sh fpga/build.tcl
#
# Adapted to the repository layout from the original direct_link_20261031/build.tcl:
# the shell changes into rtl/ so that $readmemh("weights/*.hex") in ult_context_core.sv
# resolves to rtl/weights/. Gowin writes its outputs to rtl/impl/ (git-ignored).
# NOTE: this adapted script has not been re-run after the reorganisation; the archived
# reports in fpga/reports/direct_board/ come from the original layout.
set here [file normalize [file dirname [info script]]]
cd [file join $here .. rtl]
set top cnn_direct_board
if {[info exists ::env(ULT_UART_TOP)]} {set top $::env(ULT_UART_TOP)}
if {$top ne "cnn_direct_board"} {error "Unknown top"}
set_device -name GW2AR-18C GW2AR-LV18QN88C8/I7
foreach f [glob *.v *.sv] {
    if {![string match "*_tb.*" $f]} {add_file -type verilog $f}
}
add_file -type cst [file join $here constraints board.cst]
add_file -type sdc [file join $here constraints board.sdc]
set_option -top_module $top
set_option -verilog_std sysv2017
set_option -include_path {.}
set_option -output_base_name $top
set_option -bit_security 0
run all
