`timescale 1ns/1ps
// Arithmetic reference with synchronous single read/MAC scheduling.
// Input: 256 legacy uint8 bytes, then 4096 full-context uint8 bytes.
// Camera/preprocessing/CDC and physical UART are outside this module.
`include "quant_config.vh"
module ult_context_core (
    input wire clk, rst_n, input_start,
    input wire [7:0] pixel_data,
    input wire pixel_valid,
    output wire pixel_ready, busy,
    output reg [2:0] raw_class, class_result,
    output reg signed [31:0] context_logit_q,
    output reg context_activate, context_keep, result_valid
);
    localparam IDLE=0, LOAD=1, MAC_READ=2, MAC_EXEC=3,
               POOL_READ=4, POOL_EXEC=5, ARG_INIT=6, ARG=7, DONE=8;
    reg [3:0] state;
    // stage: conv1=0, conv2=1, fc1=2, fc2=3, context head=4.
    reg [2:0] stage;
    reg context_pass;
    reg [12:0] n;
    reg [13:0] o;
    reg [10:0] k;
    reg [2:0] arg_i;
    reg signed [15:0] legacy_img[0:255], context_img[0:4095];
    reg signed [15:0] c1[0:12287], p1[0:3071];
    reg signed [15:0] c2[0:6143], p2[0:1535];
    reg signed [15:0] fc1[0:11], logits[0:5];
    reg signed [15:0] conv1_w[0:26], conv1_b[0:2];
    reg signed [15:0] conv2_w[0:161], conv2_b[0:5];
    reg signed [15:0] fc1_w[0:1151] /* synthesis syn_romstyle="block_rom" */;
    reg signed [15:0] fc1_b[0:11];
    reg signed [15:0] fc2_w[0:71], fc2_b[0:5];
    reg signed [15:0] context_w[0:1535] /* synthesis syn_romstyle="block_rom" */;
    reg signed [15:0] context_b[0:0];
    reg signed [15:0] a, w, pool_max, best;
    reg [2:0] best_id;
    reg signed [63:0] acc;
    reg signed [15:0] q_legacy,q_context,q_c1,q_p1,q_c2,q_p2,q_fc1;
    reg signed [15:0] q_w1,q_w2,q_wf1,q_wf2,q_wc;
    wire signed [31:0] product = a*w;
    wire signed [63:0] total = acc + {{32{product[31]}},product};
    wire [31:0] input_q = (pixel_data * (32'd1<<`ACT_FRAC) + 32'd127)/32'd255;
    integer row, col, channel, in_channel, ky, kx, dim, area, addr, weight_addr;
    integer last_o, last_k, shift_bits, out_channel;
    reg input_inside;
    wire signed [15:0] pooled_value = (k==0 || a>pool_max) ? a : pool_max;
    wire [2:0] arg_final = logits[arg_i]>best ? arg_i[2:0] : best_id;
    wire signed [63:0] context_q64 = requant64(total, context_b[0], `CONTEXT_SHIFT, 1'b0);

    initial begin
        $readmemh("weights/conv1_w.hex",conv1_w);$readmemh("weights/conv1_b.hex",conv1_b);
        $readmemh("weights/conv2_w.hex",conv2_w);$readmemh("weights/conv2_b.hex",conv2_b);
        $readmemh("weights/fc1_w.hex",fc1_w);$readmemh("weights/fc1_b.hex",fc1_b);
        $readmemh("weights/fc2_w.hex",fc2_w);$readmemh("weights/fc2_b.hex",fc2_b);
        $readmemh("weights/context_w.hex",context_w);$readmemh("weights/context_b.hex",context_b);
    end
    assign pixel_ready=(state==LOAD);
    assign busy=(state!=IDLE);

    // Independent registered memory outputs allow BSRAM inference. No reset
    // touches these RAM output registers; the FSM consumes only loaded data.
    always @(posedge clk)begin
        q_legacy<=legacy_img[addr[7:0]];q_context<=context_img[addr[11:0]];
        q_c1<=c1[addr[13:0]];q_p1<=p1[addr[11:0]];
        q_c2<=c2[addr[12:0]];q_p2<=p2[addr[10:0]];q_fc1<=fc1[addr[3:0]];
        q_w1<=conv1_w[weight_addr[4:0]];q_w2<=conv2_w[weight_addr[7:0]];
        q_wf1<=fc1_w[weight_addr[10:0]];q_wf2<=fc2_w[weight_addr[6:0]];
        q_wc<=context_w[weight_addr[10:0]];
    end
    always @* begin
        a=0;w=0;
        if(state==MAC_EXEC)case(stage)
            0:begin a=input_inside?(context_pass?q_context:q_legacy):16'sd0;w=q_w1;end
            1:begin a=input_inside?q_p1:16'sd0;w=q_w2;end
            2:begin a=q_p2;w=q_wf1;end
            3:begin a=q_fc1;w=q_wf2;end
            4:begin a=q_p2;w=q_wc;end
            default:begin end
        endcase
        else if(state==POOL_EXEC)a=(stage==0)?q_c1:q_c2;
    end

    function automatic signed [63:0] requant64(input signed [63:0] value,
        input signed [15:0] bias, input integer shift, input do_relu);
        reg signed [63:0] q, half_lsb;
        begin
            half_lsb=64'sd1<<(shift-1);
            if(value>=0) q=(value+half_lsb)>>>shift;
            else q=-((-value+half_lsb)>>>shift);
            q=q+{{48{bias[15]}},bias};
            if(do_relu && q<0)q=0;
            requant64=q;
        end
    endfunction
    function automatic signed [15:0] act16(input signed [63:0] value,
        input signed [15:0] bias, input integer shift, input do_relu);
        reg signed [63:0] q;
        begin
            q=requant64(value,bias,shift,do_relu);
            if(q>32767)act16=16'sh7fff;
            else if(q< -32768)act16=16'sh8000;
            else act16=q[15:0];
        end
    endfunction

    // Power-of-two address arithmetic; no variable image-size divider.
    always @* begin
        row=0;col=0;channel=0;in_channel=0;ky=0;kx=0;dim=0;area=0;
        addr=0;weight_addr=0;last_o=0;last_k=0;shift_bits=0;out_channel=0;input_inside=1;
        if(state==MAC_READ || state==MAC_EXEC) begin
            case(stage)
                0: begin
                    if(context_pass)begin row=(o&4095)>>6;col=o&63;out_channel=o>>12;dim=64;last_o=12287;end
                    else begin row=(o&255)>>4;col=o&15;out_channel=o>>8;dim=16;last_o=767;end
                    ky=k/3-1;kx=k%3-1;
                    input_inside=(row+ky>=0 && row+ky<dim && col+kx>=0 && col+kx<dim);
                    addr=context_pass ? ((row+ky)<<6)+col+kx : ((row+ky)<<4)+col+kx;
                    weight_addr=out_channel*9+k;last_k=8;shift_bits=`CONV1_SHIFT;
                end
                1: begin
                    if(context_pass)begin row=(o&1023)>>5;col=o&31;out_channel=o>>10;dim=32;area=1024;last_o=6143;end
                    else begin row=(o&63)>>3;col=o&7;out_channel=o>>6;dim=8;area=64;last_o=383;end
                    in_channel=k/9;ky=(k%9)/3-1;kx=k%3-1;
                    input_inside=(row+ky>=0 && row+ky<dim && col+kx>=0 && col+kx<dim);
                    if(context_pass)addr=(in_channel<<10)+((row+ky)<<5)+col+kx;
                    else addr=(in_channel<<6)+((row+ky)<<3)+col+kx;
                    weight_addr=out_channel*27+k;last_k=26;shift_bits=`CONV2_SHIFT;
                end
                2: begin addr=k;weight_addr=o*96+k;last_k=95;last_o=11;shift_bits=`FC1_SHIFT;end
                3: begin addr=k;weight_addr=o*12+k;last_k=11;last_o=5;shift_bits=`FC2_SHIFT;end
                4: begin addr=k;weight_addr=k;last_k=1535;last_o=0;shift_bits=`CONTEXT_SHIFT;end
                default:begin end
            endcase
        end else if(state==POOL_READ || state==POOL_EXEC)begin
            if(stage==0)begin
                if(context_pass)begin row=(o&1023)>>5;col=o&31;channel=o>>10;addr=(channel<<12)+(row<<7)+(col<<1)+(k/2)*64+k%2;last_o=3071;end
                else begin row=(o&63)>>3;col=o&7;channel=o>>6;addr=(channel<<8)+(row<<5)+(col<<1)+(k/2)*16+k%2;last_o=191;end
            end else begin
                if(context_pass)begin row=(o&255)>>4;col=o&15;channel=o>>8;addr=(channel<<10)+(row<<6)+(col<<1)+(k/2)*32+k%2;last_o=1535;end
                else begin row=(o&15)>>2;col=o&3;channel=o>>4;addr=(channel<<6)+(row<<4)+(col<<1)+(k/2)*8+k%2;last_o=95;end
            end
        end
    end

    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            state<=IDLE;stage<=0;context_pass<=0;n<=0;o<=0;k<=0;arg_i<=0;
            acc<=0;pool_max<=0;best<=0;best_id<=0;
            raw_class<=0;class_result<=0;context_logit_q<=0;
            context_activate<=0;context_keep<=0;result_valid<=0;
        end else begin
            result_valid<=0;
            case(state)
                IDLE:if(input_start)begin state<=LOAD;n<=0;end
                LOAD:if(pixel_valid)begin
                    if(n<256)legacy_img[n]<=input_q[15:0];else context_img[n-256]<=input_q[15:0];
                    if(n==4351)begin state<=MAC_READ;stage<=0;context_pass<=0;o<=0;k<=0;acc<=0;end
                    else n<=n+13'd1;
                end
                MAC_READ:state<=MAC_EXEC;
                MAC_EXEC:begin
                    if(k==last_k)begin
                        case(stage)
                            0:c1[o]<=act16(total,conv1_b[out_channel],`CONV1_SHIFT,1'b1);
                            1:c2[o]<=act16(total,conv2_b[out_channel],`CONV2_SHIFT,1'b1);
                            2:fc1[o]<=act16(total,fc1_b[o],`FC1_SHIFT,1'b1);
                            3:logits[o]<=act16(total,fc2_b[o],`FC2_SHIFT,1'b0);
                            4:begin
                                if(context_q64>64'sd2147483647)context_logit_q<=32'sh7fffffff;
                                else if(context_q64< -64'sd2147483648)context_logit_q<=32'sh80000000;
                                else context_logit_q<=context_q64[31:0];
                            end
                            default:begin end
                        endcase
                        k<=0;acc<=0;
                        if(o==last_o)begin
                            o<=0;
                            case(stage)
                                0,1:begin state<=POOL_READ;pool_max<=0;end
                                2:begin stage<=3;state<=MAC_READ;end
                                3:state<=ARG_INIT;
                                4:state<=DONE;
                                default:state<=IDLE;
                            endcase
                        end else begin o<=o+14'd1;state<=MAC_READ;end
                    end else begin acc<=total;k<=k+11'd1;state<=MAC_READ;end
                end
                POOL_READ:state<=POOL_EXEC;
                POOL_EXEC:begin
                    if(k==3)begin
                        if(stage==0)p1[o]<=pooled_value;else p2[o]<=pooled_value;
                        k<=0;pool_max<=0;
                        if(o==last_o)begin
                            o<=0;acc<=0;state<=MAC_READ;
                            if(stage==0)stage<=1;
                            else if(context_pass)stage<=4;else stage<=2;
                        end else begin o<=o+14'd1;state<=POOL_READ;end
                    end else begin pool_max<=pooled_value;k<=k+11'd1;state<=POOL_READ;end
                end
                ARG_INIT:begin arg_i<=0;best<=16'sh8000;best_id<=0;state<=ARG;end
                ARG:begin
                    if(logits[arg_i]>best)begin best<=logits[arg_i];best_id<=arg_i[2:0];end
                    if(arg_i==5)begin
                        raw_class<=arg_final;context_pass<=1;stage<=0;o<=0;k<=0;acc<=0;state<=MAC_READ;
                    end else arg_i<=arg_i+3'd1;
                end
                DONE:begin
                    context_activate<=(context_logit_q>=`CONTEXT_ACTIVATE_Q);
                    context_keep<=(context_logit_q>=`CONTEXT_KEEP_Q);
                    class_result<=(context_logit_q>=`CONTEXT_ACTIVATE_Q)?3'd5:raw_class;
                    result_valid<=1;state<=IDLE;
                end
                default:state<=IDLE;
            endcase
        end
    end
endmodule
