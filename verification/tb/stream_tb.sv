`timescale 1ns/1ps
module stream_tb;
    reg clk=0,rst_n=0,rx_valid=0,rx_error=0,tx_ready=1,tx_busy=0;
    reg [7:0] rx_data=0;
    wire [7:0] tx_data;
    wire tx_valid,waiting,fault;
    wire [15:0] completed,crc_errors,header_errors,timeouts;
    reg [7:0] packet[0:4369],expected[0:35],received[0:35];
    integer count=0,checked=0,i,stall_ticks=0;
    reg [15:0] crc;
    reg [31:0] timing;
    cnn_stream_engine #(.BYTE_TIMEOUT_CYCLES(200)) dut(.*);
    always #25 clk=~clk;
    function automatic [15:0] crc_update(input [15:0] old_crc,input [7:0] value);
        reg [15:0] c;integer j;
        begin c=old_crc^{value,8'd0};for(j=0;j<8;j=j+1)c=c[15]?(c<<1)^16'h1021:(c<<1);crc_update=c;end
    endfunction
    task send_byte(input [7:0] value);
        begin @(negedge clk);rx_data=value;rx_valid=1;@(negedge clk);rx_valid=0;end
    endtask
    task send_packet;
        begin
            for(i=0;i<4370;i=i+1)begin
                send_byte(packet[i]);
                if(i%83==0)repeat(4)@(negedge clk);
            end
        end
    endtask
    task await_result;
        begin
            wait(count==36);@(negedge clk);
            for(i=0;i<36;i=i+1)begin
                if((i<12 || (i>=16 && i<24) || (i>=28 && i<34)) && received[i]!==expected[i])
                    $fatal(1,"Packet byte %0d got %0h expected %0h",i,received[i],expected[i]);
            end
            timing={received[15],received[14],received[13],received[12]};
            if(timing<31600 || timing>31630)$fatal(1,"Incorrect inference timing %0d",timing);
            crc=16'hffff;for(i=0;i<34;i=i+1)crc=crc_update(crc,received[i]);
            if({received[35],received[34]}!==crc)$fatal(1,"Output CRC mismatch");
            checked=checked+1;
            $display("PASS stream frame=%0d raw=%0d context=%0h us=%0d",checked,received[20],{received[31],received[30],received[29],received[28]},timing);
            wait(waiting);repeat(4)@(negedge clk);count=0;
        end
    endtask
    always @(posedge clk)begin
        if(rst_n && tx_valid && tx_ready)begin
            if(count>=36)$fatal(1,"Unexpected extra response");
            received[count]=tx_data;count=count+1;
        end
    end
    always @(negedge clk)begin
        stall_ticks=stall_ticks+1;
        // Exercise formatter backpressure, independent of the CNN state.
        tx_ready=(stall_ticks%17>4);
    end
    initial begin
        repeat(4)@(negedge clk);rst_n=1;
        send_byte(8'h11);send_byte(8'ha5);send_byte(8'ha5);send_byte(8'h00);
        $readmemh("test_data/crossing_activate_input.hex",packet);
        // Wrong preprocessing ID must be rejected before the pixels are accepted.
        for(i=0;i<16;i=i+1)send_byte(i==14?8'hff:packet[i]);
        repeat(220)@(negedge clk);
        if(completed!==0 || header_errors==0)$fatal(1,"Bad header accepted");
        // A truncated header must time out and permit a fresh packet.
        send_byte(8'ha5);send_byte(8'h5a);send_byte(1);
        repeat(220)@(negedge clk);
        if(timeouts==0)$fatal(1,"Missing timeout");
        packet[100]=packet[100]^8'h01;send_packet();
        repeat(10)@(negedge clk);
        if(crc_errors!==1 || completed!==0 || count!=0)$fatal(1,"Bad CRC accepted");
        packet[100]=packet[100]^8'h01;
        $readmemh("test_data/crossing_activate_expected.hex",expected);
        send_packet();
        // Noise while computing must not overwrite the committed input buffer.
        repeat(20)send_byte(8'ha5);
        await_result();
        $readmemh("test_data/class_straight_input.hex",packet);
        $readmemh("test_data/class_straight_expected.hex",expected);
        send_packet();await_result();
        $readmemh("test_data/class_sharp_right_input.hex",packet);
        $readmemh("test_data/class_sharp_right_expected.hex",expected);
        send_packet();await_result();
        if(completed!==3 || fault!==0)$fatal(1,"Final completion status");
        $display("PASS CRC/header/timeout recovery, input stalls, busy isolation, 3 distinct frames and output stalls");
        $finish;
    end
    initial begin #130000000;$fatal(1,"Global watchdog");end
endmodule
