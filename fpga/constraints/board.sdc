create_clock -name clk_27m -period 37.037037 [get_ports {clk_27m}]
create_generated_clock -name core_20m -source [get_ports {clk_27m}] -multiply_by 20 -divide_by 27 [get_pins {u_clock/u_pll/CLKOUTD}]
# The UART pin is asynchronous; only the first synchronizer FF is exempted.
set_false_path -from [get_ports {xiao_rx}]
