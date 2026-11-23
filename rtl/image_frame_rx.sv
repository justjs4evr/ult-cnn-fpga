`timescale 1ns/1ps
// Prepared-image protocol v1: A5 5A, 16-byte header, 4352 pixels, CRC16 LE.
// A frame is exposed only after CRC validation. RAM is never reset/cleared.
module image_frame_rx #(
    parameter integer BYTE_TIMEOUT_CYCLES=2000000
)(
    input wire clk,rst_n,enable,
    input wire [7:0] byte_data,
    input wire byte_valid,byte_error,
    input wire [31:0] now_us,
    input wire [12:0] read_addr,
    output reg [7:0] read_data,
    output reg frame_valid,
    output reg [31:0] frame_id,capture_tick,first_byte_us,
    output reg [15:0] crc_errors,header_errors,timeouts
);
    localparam SEEK_A5=0,SEEK_5A=1,HEADER=2,PIXELS=3,CRC_LO=4,CRC_HI=5;
    reg [2:0] state;
    reg [4:0] hindex;
    reg [12:0] pindex;
    reg [15:0] crc;
    reg [7:0] crc_low;
    reg [20:0] idle_cycles;
    reg [7:0] pixels[0:4351];
    reg header_ok;
    function automatic [15:0] crc_update(input [15:0] old_crc,input [7:0] value);
        reg [15:0] c;integer i;
        begin
            c=old_crc^{value,8'h00};
            for(i=0;i<8;i=i+1)c=c[15]?(c<<1)^16'h1021:(c<<1);
            crc_update=c;
        end
    endfunction
    always @(posedge clk)read_data<=pixels[read_addr];
    always @*begin
        header_ok=1;
        case(hindex)
            2:header_ok=(byte_data==1);
            3:header_ok=(byte_data==8'h10);
            12:header_ok=(byte_data==8'h00);
            13:header_ok=(byte_data==8'h11);
            14:header_ok=(byte_data==8'h01);
            15:header_ok=(byte_data==8'h02);
            default:begin end
        endcase
    end
    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            state<=SEEK_A5;hindex<=0;pindex<=0;crc<=16'hffff;crc_low<=0;idle_cycles<=0;
            frame_valid<=0;frame_id<=0;capture_tick<=0;first_byte_us<=0;
            crc_errors<=0;header_errors<=0;timeouts<=0;
        end else begin
            frame_valid<=0;
            if(!enable)begin state<=SEEK_A5;idle_cycles<=0;end
            else if(byte_error)begin state<=SEEK_A5;idle_cycles<=0;header_errors<=header_errors+16'd1;end
            else if(byte_valid)begin
                idle_cycles<=0;
                case(state)
                    SEEK_A5:if(byte_data==8'ha5)begin
                        state<=SEEK_5A;crc<=crc_update(16'hffff,byte_data);first_byte_us<=now_us;
                    end
                    SEEK_5A:if(byte_data==8'h5a)begin state<=HEADER;hindex<=2;crc<=crc_update(crc,byte_data);end
                        else if(byte_data==8'ha5)begin crc<=crc_update(16'hffff,byte_data);first_byte_us<=now_us;end
                        else state<=SEEK_A5;
                    HEADER:if(!header_ok)begin
                        header_errors<=header_errors+16'd1;
                        if(byte_data==8'ha5)begin state<=SEEK_5A;crc<=crc_update(16'hffff,byte_data);first_byte_us<=now_us;end
                        else state<=SEEK_A5;
                    end else begin
                        crc<=crc_update(crc,byte_data);
                        case(hindex)
                            4:frame_id[7:0]<=byte_data;
                            5:frame_id[15:8]<=byte_data;
                            6:frame_id[23:16]<=byte_data;
                            7:frame_id[31:24]<=byte_data;
                            8:capture_tick[7:0]<=byte_data;
                            9:capture_tick[15:8]<=byte_data;
                            10:capture_tick[23:16]<=byte_data;
                            11:capture_tick[31:24]<=byte_data;
                            default:begin end
                        endcase
                        if(hindex==15)begin state<=PIXELS;pindex<=0;end
                        else hindex<=hindex+5'd1;
                    end
                    PIXELS:begin
                        pixels[pindex]<=byte_data;crc<=crc_update(crc,byte_data);
                        if(pindex==4351)state<=CRC_LO;else pindex<=pindex+13'd1;
                    end
                    CRC_LO:begin crc_low<=byte_data;state<=CRC_HI;end
                    CRC_HI:begin
                        if({byte_data,crc_low}==crc)frame_valid<=1;
                        else crc_errors<=crc_errors+16'd1;
                        state<=SEEK_A5;
                    end
                    default:state<=SEEK_A5;
                endcase
            end else if(state!=SEEK_A5)begin
                if(idle_cycles>=BYTE_TIMEOUT_CYCLES-1)begin state<=SEEK_A5;idle_cycles<=0;timeouts<=timeouts+16'd1;end
                else idle_cycles<=idle_cycles+21'd1;
            end
        end
    end
endmodule
