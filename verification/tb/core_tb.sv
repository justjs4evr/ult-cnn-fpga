`timescale 1ns/1ps
`include "quant_config.vh"
module core_tb;
    reg clk=0,rst_n=0,input_start=0,pixel_valid=0;
    reg [7:0] pixel_data=0;
    wire pixel_ready,busy,result_valid,context_activate,context_keep;
    wire [2:0] raw_class,class_result;
    wire signed [31:0] context_logit_q;
    reg [7:0] legacy_bytes[0:255],context_bytes[0:4095],expected_raw[0:0],expected_class[0:0];
    reg signed [15:0] lc1[0:767],lp1[0:191],lc2[0:383],lp2[0:95],lf1[0:11],ll[0:5];
    reg signed [15:0] cc1[0:12287],cp1[0:3071],cc2[0:6143],cp2[0:1535];
    reg signed [31:0] cg[0:0];
    integer i,cycles,frame,discard;
    string vector_dir;
    ult_context_core dut(.*);
    always #5 clk=~clk;
    initial begin
        if(!$value$plusargs("VECTOR=%s",vector_dir))$fatal(1,"Missing VECTOR");
        $readmemh({vector_dir,"/legacy_input.hex"},legacy_bytes);
        $readmemh({vector_dir,"/context_input.hex"},context_bytes);
        $readmemh({vector_dir,"/legacy_conv1.hex"},lc1);$readmemh({vector_dir,"/legacy_pool1.hex"},lp1);
        $readmemh({vector_dir,"/legacy_conv2.hex"},lc2);$readmemh({vector_dir,"/legacy_pool2.hex"},lp2);
        $readmemh({vector_dir,"/legacy_fc1.hex"},lf1);$readmemh({vector_dir,"/legacy_logits.hex"},ll);
        $readmemh({vector_dir,"/context_conv1.hex"},cc1);$readmemh({vector_dir,"/context_pool1.hex"},cp1);
        $readmemh({vector_dir,"/context_conv2.hex"},cc2);$readmemh({vector_dir,"/context_pool2.hex"},cp2);
        $readmemh({vector_dir,"/context_logit.hex"},cg);
        $readmemh({vector_dir,"/raw_class.hex"},expected_raw);$readmemh({vector_dir,"/class_result.hex"},expected_class);
        repeat(3)@(negedge clk);rst_n=1;
        // Two consecutive frames verify reuse without resetting internal RAM.
        for(frame=0;frame<2;frame=frame+1)begin
            input_start=1;@(negedge clk);input_start=0;
            for(i=0;i<4352;i=i+1)begin
                if(!pixel_ready)$fatal(1,"Unexpected backpressure while loading");
                if(i%37==0)begin pixel_valid=0;@(negedge clk);end
                pixel_valid=1;pixel_data=(i<256)?legacy_bytes[i]:context_bytes[i-256];
                @(negedge clk);
            end
            pixel_valid=0;cycles=0;
            while(dut.state!=6 && cycles<700000)begin @(negedge clk);cycles=cycles+1;end
            if(dut.state!=6)$fatal(1,"Legacy timeout");
            for(i=0;i<768;i=i+1)if(dut.c1[i]!==lc1[i])$fatal(1,"legacy conv1 %0d got %0d expected %0d",i,dut.c1[i],lc1[i]);
            for(i=0;i<192;i=i+1)if(dut.p1[i]!==lp1[i])$fatal(1,"legacy pool1 %0d",i);
            for(i=0;i<384;i=i+1)if(dut.c2[i]!==lc2[i])$fatal(1,"legacy conv2 %0d",i);
            for(i=0;i<96;i=i+1)if(dut.p2[i]!==lp2[i])$fatal(1,"legacy pool2 %0d",i);
            for(i=0;i<12;i=i+1)if(dut.fc1[i]!==lf1[i])$fatal(1,"legacy fc1 %0d",i);
            for(i=0;i<6;i=i+1)if(dut.logits[i]!==ll[i])$fatal(1,"legacy logits %0d",i);
            while(!result_valid && cycles<700000)begin @(negedge clk);cycles=cycles+1;end
            if(!result_valid)$fatal(1,"Context timeout");
            for(i=0;i<12288;i=i+1)if(dut.c1[i]!==cc1[i])$fatal(1,"context conv1 %0d got %0d expected %0d",i,dut.c1[i],cc1[i]);
            for(i=0;i<3072;i=i+1)if(dut.p1[i]!==cp1[i])$fatal(1,"context pool1 %0d",i);
            for(i=0;i<6144;i=i+1)if(dut.c2[i]!==cc2[i])$fatal(1,"context conv2 %0d got %0d expected %0d",i,dut.c2[i],cc2[i]);
            for(i=0;i<1536;i=i+1)if(dut.p2[i]!==cp2[i])$fatal(1,"context pool2 %0d",i);
            if(context_logit_q!==cg[0])$fatal(1,"context logit got %0d expected %0d",context_logit_q,cg[0]);
            if(raw_class!==expected_raw[0][2:0])$fatal(1,"raw class");
            if(class_result!==expected_class[0][2:0])$fatal(1,"final class");
            if(context_activate!==(cg[0]>=`CONTEXT_ACTIVATE_Q))$fatal(1,"activate flag");
            if(context_keep!==(cg[0]>=`CONTEXT_KEEP_Q))$fatal(1,"keep flag");
            $display("PASS vector=%s frame=%0d cycles=%0d raw=%0d class=%0d context=%0d",vector_dir,frame,cycles,raw_class,class_result,context_logit_q);
            @(negedge clk);if(result_valid)$fatal(1,"result_valid not a pulse");
        end
        $finish;
    end
    initial begin #15000000;$fatal(1,"Global timeout");end
endmodule
