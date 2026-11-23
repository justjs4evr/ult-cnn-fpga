`timescale 1ns/1ps
module uart_8n1 #(
    parameter integer CLK_HZ=20000000,
    parameter integer BAUD=460800
)(
    input wire clk,rst_n,rx_pin,
    output reg [7:0] rx_data,
    output reg rx_valid,rx_error,
    input wire [7:0] tx_data,
    input wire tx_valid,
    output wire tx_ready,
    output reg tx_pin,
    output wire tx_busy
);
    localparam integer DIV=(CLK_HZ+BAUD/2)/BAUD;
    reg rx_meta,rx_sync;
    reg [1:0] rx_state;
    reg [15:0] rx_count,tx_count;
    reg [2:0] rx_bit;
    reg [7:0] rx_shift;
    reg [9:0] tx_shift;
    reg [3:0] tx_bit;
    reg sending;
    assign tx_ready=!sending;
    assign tx_busy=sending;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n)begin rx_meta<=1;rx_sync<=1;end
        else begin rx_meta<=rx_pin;rx_sync<=rx_meta;end
    end
    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            rx_state<=0;rx_count<=0;rx_bit<=0;rx_shift<=0;
            rx_data<=0;rx_valid<=0;rx_error<=0;
        end else begin
            rx_valid<=0;rx_error<=0;
            case(rx_state)
                0:if(!rx_sync)begin rx_count<=DIV/2-1;rx_state<=1;end
                1:if(rx_count==0)begin
                    if(!rx_sync)begin rx_state<=2;rx_count<=DIV-1;rx_bit<=0;end
                    else rx_state<=0;
                end else rx_count<=rx_count-16'd1;
                2:if(rx_count==0)begin
                    rx_shift[rx_bit]<=rx_sync;rx_count<=DIV-1;
                    if(rx_bit==7)rx_state<=3;else rx_bit<=rx_bit+3'd1;
                end else rx_count<=rx_count-16'd1;
                3:if(rx_count==0)begin
                    rx_state<=0;
                    if(rx_sync)begin rx_data<=rx_shift;rx_valid<=1;end
                    else rx_error<=1;
                end else rx_count<=rx_count-16'd1;
                default:rx_state<=0;
            endcase
        end
    end
    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin tx_pin<=1;sending<=0;tx_count<=0;tx_bit<=0;tx_shift<=10'h3ff;end
        else if(!sending)begin
            tx_pin<=1;
            if(tx_valid)begin
                tx_shift<={1'b1,tx_data,1'b0};tx_bit<=0;tx_count<=DIV-1;tx_pin<=0;sending<=1;
            end
        end else if(tx_count==0)begin
            if(tx_bit==9)begin tx_pin<=1;sending<=0;end
            else begin tx_shift<={1'b1,tx_shift[9:1]};tx_pin<=tx_shift[1];tx_bit<=tx_bit+4'd1;tx_count<=DIV-1;end
        end else tx_count<=tx_count-16'd1;
    end
endmodule
