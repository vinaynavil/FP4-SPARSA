`timescale 1ns / 1ps
// ============================================================
// Testbench : tb_weight_bram
// Tests BRAM weight buffer write via AXI and read-back
// through weight_bram Port B.
// ============================================================
module tb_weight_bram;

    reg clk = 0;
    reg aresetn = 0;
    always #1.43 clk = ~clk;

    // AXI signals
    reg  [4:0]  awaddr;
    reg         awvalid;
    wire        awready;
    reg  [31:0] wdata;
    reg  [3:0]  wstrb;
    reg         wvalid;
    wire        wready;
    wire [1:0]  bresp;
    wire        bvalid;
    reg         bready;
    reg  [4:0]  araddr;
    reg         arvalid;
    wire        arready;
    wire [31:0] rdata;
    wire [1:0]  rresp;
    wire        rvalid;
    reg         rready;

    // Dummy status
    reg        valid_out_i      = 0;
    reg [6:0]  zero_skip_count_i = 0;
    reg [5:0]  sparse_act_count_i = 0;
    reg [5:0]  sparse_wgt_count_i = 0;
    reg [3:0]  sat_flags_i      = 0;

    wire sparse_en, mode, bank_switch;
    wire bram_ena;
    wire [2:0]  bram_addr;
    wire [127:0] bram_din;

    // AXI ctrl DUT
    axi_lite_ctrl u_ctrl (
        .s_axi_aclk       (clk),
        .s_axi_aresetn    (aresetn),
        .s_axi_awaddr     (awaddr),
        .s_axi_awvalid    (awvalid),
        .s_axi_awready    (awready),
        .s_axi_wdata      (wdata),
        .s_axi_wstrb      (wstrb),
        .s_axi_wvalid     (wvalid),
        .s_axi_wready     (wready),
        .s_axi_bresp      (bresp),
        .s_axi_bvalid     (bvalid),
        .s_axi_bready     (bready),
        .s_axi_araddr     (araddr),
        .s_axi_arvalid    (arvalid),
        .s_axi_arready    (arready),
        .s_axi_rdata      (rdata),
        .s_axi_rresp      (rresp),
        .s_axi_rvalid     (rvalid),
        .s_axi_rready     (rready),
        .valid_out        (valid_out_i),
        .zero_skip_count  (zero_skip_count_i),
        .sparse_act_count (sparse_act_count_i),
        .sparse_wgt_count (sparse_wgt_count_i),
        .sat_flags        (sat_flags_i),
        .sparse_en        (sparse_en),
        .mode             (mode),
        .bank_switch      (bank_switch),
        .bram_ena         (bram_ena),
        .bram_addr        (bram_addr),
        .bram_din         (bram_din)
    );

    // BRAM DUT
    wire [127:0] bram_dout;
    reg  [2:0]   bram_raddr = 0;

    weight_bram u_bram (
        .clk    (clk),
        .ena_a  (bram_ena),
        .addr_a (bram_addr),
        .din_a  (bram_din),
        .addr_b (bram_raddr),
        .dout_b (bram_dout)
    );

    // AXI write task
    task axi_write;
        input [4:0]  addr;
        input [31:0] data;
        begin
            @(posedge clk);
            awaddr = addr; awvalid = 1;
            wdata  = data; wstrb = 4'hF; wvalid = 1;
            bready = 1;
            @(posedge clk);
            while (!awready) @(posedge clk);
            awvalid = 0;
            while (!wready)  @(posedge clk);
            wvalid = 0;
            while (!bvalid)  @(posedge clk);
            @(posedge clk);
            bready = 0;
        end
    endtask

    integer errors = 0;

    initial begin
        awaddr=0; awvalid=0; wdata=0; wstrb=0; wvalid=0; bready=0;
        araddr=0; arvalid=0; rready=0;

        repeat(4) @(posedge clk);
        aresetn = 1;
        repeat(2) @(posedge clk);

        // ---- TC1: Set WADDR = 0 ----
        $display("TC1: Set WADDR=0");
        axi_write(5'h08, 32'd0);
        $display("  PASS");

        // ---- TC2: Write word 0 to BRAM ----
        // WDATA_LO lower 32: 0xDEADBEEF
        // WDATA_LO upper 32: 0xCAFEBABE
        // WDATA_HI        : 0x12345678 -> triggers write at addr 0
        $display("TC2: Write BRAM word 0");
        axi_write(5'h0C, 32'hDEADBEEF);  // lo[31:0]
        // lo[63:32] — write again to same addr (upper half)
        // For simplicity this TB uses the lower 32 bits only
        axi_write(5'h10, 32'h12345678);  // hi -> triggers BRAM write

        // Wait for BRAM write to propagate
        repeat(3) @(posedge clk);

        // Read back from BRAM Port B at addr 0
        bram_raddr = 3'd0;
        repeat(2) @(posedge clk);  // BRAM registered output latency

        $display("  BRAM dout[127:96] = 0x%08X (expect 0x12345678)", bram_dout[127:96]);
        if (bram_dout[127:96] !== 32'h12345678) begin
            $display("  FAIL"); errors = errors + 1;
        end else
            $display("  PASS");

        // ---- TC3: Verify WADDR auto-incremented to 1 ----
        $display("TC3: Check WADDR auto-increment");
        // Write another word — should go to addr 1
        axi_write(5'h0C, 32'hAABBCCDD);
        axi_write(5'h10, 32'h11223344);
        repeat(3) @(posedge clk);
        bram_raddr = 3'd1;
        repeat(2) @(posedge clk);
        $display("  BRAM[1][127:96] = 0x%08X (expect 0x11223344)", bram_dout[127:96]);
        if (bram_dout[127:96] !== 32'h11223344) begin
            $display("  FAIL"); errors = errors + 1;
        end else
            $display("  PASS");

        // ---- TC4: CTRL register still works ----
        $display("TC4: CTRL write sparse_en=1");
        axi_write(5'h00, 32'h00000001);
        if (sparse_en !== 1) begin
            $display("  FAIL"); errors = errors + 1;
        end else
            $display("  PASS");

        repeat(4) @(posedge clk);
        if (errors == 0)
            $display("\n*** ALL TESTS PASSED ***");
        else
            $display("\n*** %0d TEST(S) FAILED ***", errors);
        $finish;
    end

    initial begin #100000; $display("TIMEOUT"); $finish; end

endmodule
