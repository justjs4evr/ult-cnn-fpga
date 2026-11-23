`timescale 1ns/1ps
// Version 2, 36-byte framing with CRC16-CCITT-FALSE.
// Connect byte_data/byte_valid/byte_ready to a separate UART 8N1 transmitter.
module result_packet_tx(
    input wire clk,rst_n,packet_start,
    input wire [31:0] frame_id,capture_tick,inference_us,model_id,capture_to_tx_us,
    input wire [2:0] raw_class,
    input wire [7:0] flags,
    input wire [15:0] preprocess_id,
    input wire signed [31:0] context_logit_q,
    output wire [7:0] byte_data,
    output wire byte_valid,
    input wire byte_ready,
    output reg busy,packet_done
);
    reg [271:0] payload;
    reg [5:0] index;
    reg [15:0] crc;
    function automatic [15:0] update_crc(input [15:0] old_crc,input [7:0] value);
        reg [15:0] c;integer j;
        begin
            c=old_crc^{value,8'd0};
            for(j=0;j<8;j=j+1)c=c[15]?(c<<1)^16'h1021:(c<<1);
            update_crc=c;
        end
    endfunction
    assign byte_data=(index<34)?payload[index*8+:8]:(index==34?crc[7:0]:crc[15:8]);
    assign byte_valid=busy;
    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin payload<=0;index<=0;crc<=16'hffff;busy<=0;packet_done<=0;end
        else begin
            packet_done<=0;
            if(!busy)begin
                if(packet_start)begin
                    payload[15:0]<=16'h55aa;payload[23:16]<=8'd2;payload[31:24]<=8'd1;
                    payload[63:32]<=frame_id;payload[95:64]<=capture_tick;
                    payload[127:96]<=inference_us;payload[159:128]<=model_id;
                    payload[167:160]<={5'd0,raw_class};payload[175:168]<=flags;
                    payload[191:176]<=preprocess_id;payload[223:192]<=capture_to_tx_us;
                    payload[255:224]<=context_logit_q;payload[271:256]<=0;
                    index<=0;crc<=16'hffff;busy<=1;
                end
            end else if(byte_ready)begin
                if(index<34)crc<=update_crc(crc,byte_data);
                if(index==35)begin busy<=0;packet_done<=1;end
                else index<=index+1;
            end
        end
    end
endmodule
