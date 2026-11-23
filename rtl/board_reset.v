`timescale 1ns/1ps
// Asynchronous assertion on lost lock, synchronous release after 256 core clocks.
module board_reset(input wire clk, locked, output wire rst_n);
    reg [8:0] startup = 0;
    always @(posedge clk or negedge locked) begin
        if (!locked) startup <= 0;
        else if (!startup[8]) startup <= startup + 9'd1;
    end
    assign rst_n = startup[8];
endmodule
