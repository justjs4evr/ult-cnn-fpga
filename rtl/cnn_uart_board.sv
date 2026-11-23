`timescale 1ns/1ps
module cnn_uart_board(input wire clk_27m,uart_rx,output wire uart_tx,status_led_n);
    wire clk,locked,rst_n,rx_valid,rx_error,tx_valid,tx_ready,tx_busy,waiting,fault;
    wire [7:0] rx_data,tx_data;
    wire [15:0] completed,crc_errors,header_errors,timeouts;
    reg [24:0] blink;
    clock_20m u_clock(clk_27m,clk,locked);
    board_reset u_reset(clk,locked,rst_n);
    uart_8n1 #(.BAUD(460800)) u_uart(clk,rst_n,uart_rx,rx_data,rx_valid,rx_error,
        tx_data,tx_valid,tx_ready,uart_tx,tx_busy);
    cnn_stream_engine u_engine(clk,rst_n,rx_data,rx_valid,rx_error,tx_data,tx_valid,
        tx_ready,tx_busy,waiting,completed,fault,crc_errors,header_errors,timeouts);
    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)blink<=0;
        else if(blink==19999999)blink<=0;
        else blink<=blink+25'd1;
    end
    // Before the first valid inference: brief single pulse. After inference: double pulse.
    wire good_on=(completed!=0)?(blink<2000000 || (blink>=4000000 && blink<6000000)):(blink<1000000);
    assign status_led_n=!(rst_n && (fault || good_on));
endmodule
