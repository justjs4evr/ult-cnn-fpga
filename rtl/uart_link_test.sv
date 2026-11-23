`timescale 1ns/1ps
// Periodic identifier permits read-only COM-port discovery. Received bytes echo.
module uart_link_test(input wire clk_27m,uart_rx,output wire uart_tx,status_led_n);
    wire clk,locked,rst_n;
    wire [7:0] rx_data;
    wire rx_valid,rx_error,tx_ready,tx_busy;
    reg [7:0] fifo[0:15];
    reg [3:0] wr,rd;
    reg [4:0] count;
    reg [23:0] timer;
    reg [4:0] banner_index;
    wire [127:0] banner={"ULT_UART_READY",8'h0d,8'h0a};
    wire banner_active=(banner_index<16);
    wire [7:0] tx_data=(count!=0)?fifo[rd]:banner[(15-banner_index)*8+:8];
    wire tx_valid=(count!=0)||banner_active;
    wire push=rx_valid && count<16;
    wire pop=tx_valid && tx_ready && count!=0;
    clock_20m u_clock(clk_27m,clk,locked);
    board_reset u_reset(clk,locked,rst_n);
    uart_8n1 #(.BAUD(115200)) u_uart(clk,rst_n,uart_rx,rx_data,rx_valid,rx_error,
        tx_data,tx_valid,tx_ready,uart_tx,tx_busy);
    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin wr<=0;rd<=0;count<=0;timer<=0;banner_index<=0;end
        else begin
            if(push)begin fifo[wr]<=rx_data;wr<=wr+4'd1;end
            if(pop)rd<=rd+4'd1;
            case({push,pop})2'b10:count<=count+5'd1;2'b01:count<=count-5'd1;default:begin end endcase
            if(banner_active)begin
                timer<=0;
                if(count==0 && tx_ready)banner_index<=banner_index+5'd1;
            end else if(timer==24'd9999999)begin timer<=0;banner_index<=0;end
            else timer<=timer+24'd1;
        end
    end
    assign status_led_n=!(rst_n && timer<24'd1000000);
endmodule
