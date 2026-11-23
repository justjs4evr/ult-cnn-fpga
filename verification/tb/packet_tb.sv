`timescale 1ns/1ps
module packet_tb;
    reg clk=0,rst_n=0,packet_start=0,byte_ready=0;
    wire [7:0] byte_data;
    wire byte_valid,busy,packet_done;
    reg [7:0] expected[0:35];
    reg [31:0] frame_id,capture_tick,inference_us,model_id,capture_to_tx_us;
    reg [2:0] raw_class;reg [7:0] flags;reg [15:0] preprocess_id;
    reg signed [31:0] context_logit_q;
    integer i,tick;
    result_packet_tx dut(.*);
    always #5 clk=~clk;
    initial begin
        $readmemh("expected_uart_packet.hex",expected);
        frame_id={expected[7],expected[6],expected[5],expected[4]};
        capture_tick={expected[11],expected[10],expected[9],expected[8]};
        inference_us={expected[15],expected[14],expected[13],expected[12]};
        model_id={expected[19],expected[18],expected[17],expected[16]};
        raw_class=expected[20][2:0];flags=expected[21];preprocess_id={expected[23],expected[22]};
        capture_to_tx_us={expected[27],expected[26],expected[25],expected[24]};
        context_logit_q={expected[31],expected[30],expected[29],expected[28]};
        repeat(3)@(negedge clk);rst_n=1;packet_start=1;
        @(negedge clk);packet_start=0;i=0;tick=0;
        while(i<36 && tick<100)begin
            byte_ready=(tick%3!=0);
            if(!byte_valid)$fatal(1,"Packet ended early");
            if(byte_data!==expected[i])$fatal(1,"Packet byte %0d got %h expected %h",i,byte_data,expected[i]);
            if(byte_ready)i=i+1;
            @(negedge clk);tick=tick+1;
        end
        if(i!=36 || !packet_done || byte_valid)$fatal(1,"Packet handshake/length");
        $display("PASS packet 36 bytes including CRC and ready stalls");$finish;
    end
endmodule
