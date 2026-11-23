`timescale 1ns/1ps
// Stop-and-wait: host sends one input frame, waits for the entire result, then sends next.
module cnn_stream_engine #(
    parameter integer BYTE_TIMEOUT_CYCLES=2000000
)(
    input wire clk,rst_n,
    input wire [7:0] rx_data,
    input wire rx_valid,rx_error,
    output wire [7:0] tx_data,
    output wire tx_valid,
    input wire tx_ready,tx_busy,
    output wire waiting,
    output reg [15:0] completed,
    output reg fault,
    output wire [15:0] crc_errors,header_errors,timeouts
);
    localparam WAIT_FRAME=0,CORE_START=1,READ_WAIT=2,LOAD=3,INFER=4,PACK=5,DRAIN=6;
    reg [2:0] state;
    reg [12:0] index;
    reg [19:0] inference_cycles;
    reg [4:0] us_div,compute_div;
    reg [31:0] now_us,compute_us;
    reg [31:0] active_id,active_tick,active_first_us,measured_inference_us;
    reg [2:0] saved_raw;
    reg signed [31:0] saved_context;
    reg saved_activate,saved_keep;
    wire frame_valid;
    wire [31:0] frame_id,capture_tick,first_byte_us;
    wire [7:0] pixels;
    wire pixel_ready,core_busy,result_valid,activate,keep;
    wire [2:0] raw_class,class_result;
    wire signed [31:0] context_q;
    wire packet_busy,packet_done;
    wire [7:0] flags=8'h01 | (saved_activate?8'h10:8'h00) | (saved_keep?8'h20:8'h00);
    assign waiting=(state==WAIT_FRAME);
    image_frame_rx #(.BYTE_TIMEOUT_CYCLES(BYTE_TIMEOUT_CYCLES)) u_frame(
        clk,rst_n,waiting,rx_data,rx_valid,rx_error,now_us,index,pixels,frame_valid,
        frame_id,capture_tick,first_byte_us,crc_errors,header_errors,timeouts);
    ult_context_core u_core(
        .clk(clk),.rst_n(rst_n),.input_start(state==CORE_START),.pixel_data(pixels),
        .pixel_valid(state==LOAD),.pixel_ready(pixel_ready),.busy(core_busy),
        .raw_class(raw_class),.class_result(class_result),.context_logit_q(context_q),
        .context_activate(activate),.context_keep(keep),.result_valid(result_valid));
    result_packet_tx u_packet(
        .clk(clk),.rst_n(rst_n),.packet_start(state==PACK),
        .frame_id(active_id),.capture_tick(active_tick),.inference_us(measured_inference_us),
        .model_id(32'h26c8ea39),.capture_to_tx_us(now_us-active_first_us),
        .raw_class(saved_raw),.flags(flags),.preprocess_id(16'h0201),.context_logit_q(saved_context),
        .byte_data(tx_data),.byte_valid(tx_valid),.byte_ready(tx_ready),.busy(packet_busy),.packet_done(packet_done));
    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin now_us<=0;us_div<=0;end
        else if(us_div==19)begin us_div<=0;now_us<=now_us+32'd1;end
        else us_div<=us_div+5'd1;
    end
    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            state<=WAIT_FRAME;index<=0;inference_cycles<=0;compute_div<=0;compute_us<=0;
            active_id<=0;active_tick<=0;active_first_us<=0;measured_inference_us<=0;
            saved_raw<=0;saved_context<=0;saved_activate<=0;saved_keep<=0;
            completed<=0;fault<=0;
        end else case(state)
            WAIT_FRAME:if(frame_valid)begin
                active_id<=frame_id;active_tick<=capture_tick;active_first_us<=first_byte_us;
                state<=CORE_START;index<=0;
            end
            CORE_START:state<=READ_WAIT;
            READ_WAIT:state<=LOAD;
            LOAD:if(pixel_ready)begin
                if(index==4351)begin state<=INFER;inference_cycles<=0;compute_us<=0;compute_div<=0;end
                else begin index<=index+13'd1;state<=READ_WAIT;end
            end
            INFER:begin
                inference_cycles<=inference_cycles+20'd1;
                if(compute_div==19)begin compute_div<=0;compute_us<=compute_us+32'd1;end
                else compute_div<=compute_div+5'd1;
                if(result_valid)begin
                    saved_raw<=raw_class;saved_context<=context_q;saved_activate<=activate;saved_keep<=keep;
                    measured_inference_us<=compute_us+32'd1;completed<=completed+16'd1;state<=PACK;
                end else if(inference_cycles>=700000)begin fault<=1;state<=DRAIN;end
            end
            PACK:state<=DRAIN;
            DRAIN:if(!packet_busy && !tx_busy)state<=WAIT_FRAME;
            default:begin fault<=1;state<=WAIT_FRAME;end
        endcase
    end
endmodule
