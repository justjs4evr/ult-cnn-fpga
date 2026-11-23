`timescale 1ns/1ps
// GW2AR-18C rPLL: 27/9*40 = 120 MHz, VCO=960 MHz, CLKOUTD=120/6=20 MHz.
module clock_20m(input wire clk_27m, output wire clk_20m, output wire locked);
    rPLL #(
        .FCLKIN("27"), .DYN_IDIV_SEL("false"), .IDIV_SEL(8),
        .DYN_FBDIV_SEL("false"), .FBDIV_SEL(39),
        .DYN_ODIV_SEL("false"), .ODIV_SEL(8),
        .PSDA_SEL("0000"), .DYN_DA_EN("false"), .DUTYDA_SEL("1000"),
        .CLKFB_SEL("internal"), .CLKOUT_BYPASS("false"),
        .CLKOUTP_BYPASS("false"), .CLKOUTD_BYPASS("false"),
        .DYN_SDIV_SEL(6), .CLKOUTD_SRC("CLKOUT"),
        .CLKOUTD3_SRC("CLKOUT"), .DEVICE("GW2AR-18C")
    ) u_pll (
        .CLKOUT(), .CLKOUTP(), .CLKOUTD(clk_20m), .CLKOUTD3(), .LOCK(locked),
        .CLKIN(clk_27m), .CLKFB(1'b0), .RESET(1'b0), .RESET_P(1'b0),
        .FBDSEL(6'd0), .IDSEL(6'd0), .ODSEL(6'd0),
        .PSDA(4'd0), .DUTYDA(4'd0), .FDLY(4'd0)
    );
endmodule
