`timescale 1ns / 1ps
// ============================================================
// Testbench : tb_axi_lite_ctrl  (v19)
// Tests AXI-Lite register write/read-back for axi_lite_ctrl v2
// (BRAM weight-buffer variant).
//
// Updated vs v17 TB:
//   - Removed load_weight_lo/hi (no longer ports; pulses are
//     derived at top level from bram_ena/bram_addr)
//   - Added fifo_full/fifo_empty status inputs
//   - awaddr/araddr widened to 5 bits
//   - STATUS is at 0x04 (not 0x08); includes fifo bits [25:24]
//   - Added BRAM write-port checks: 3x WDATA_LO + WDATA_HI
//     assembles 128-bit word, pulses bram_ena, auto-increments
//   - bank_switch verified as 1-cycle pulse (auto-clear)
// ============================================================
module tb_axi_lite_ctrl;

    reg clk = 0;
    reg aresetn = 0;
    always #1.43 clk = ~clk;   // ~350 MHz

    // AXI-Lite signals
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

    // Dummy status inputs
    reg        valid_out        = 1;
    reg [6:0]  zero_skip_count  = 7'd42;
    reg [5:0]  sparse_act_count = 6'd15;
    reg [5:0]  sparse_wgt_count = 6'd7;
    reg [3:0]  sat_flags        = 4'b1010;
    reg        fifo_full        = 1'b0;
    reg        fifo_empty       = 1'b1;

    // Control / BRAM outputs
    wire         sparse_en, mode, bank_switch;
    wire         bram_ena;
    wire [2:0]   bram_addr;
    wire [127:0] bram_din;

    // bank_switch pulse monitor
    reg bsw_seen;
    always @(posedge clk) if (bank_switch) bsw_seen <= 1'b1;

    // bram_ena pulse monitor (captures commit cycle)
    reg          ena_seen;
    reg [2:0]    ena_addr;
    reg [127:0]  ena_din;
    always @(posedge clk) begin
        if (bram_ena) begin
            ena_seen <= 1'b1;
            ena_addr <= bram_addr;
            ena_din  <= bram_din;
        end
    end

    axi_lite_ctrl dut (
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
        .valid_out        (valid_out),
        .zero_skip_count  (zero_skip_count),
        .sparse_act_count (sparse_act_count),
        .sparse_wgt_count (sparse_wgt_count),
        .sat_flags        (sat_flags),
        .fifo_full        (fifo_full),
        .fifo_empty       (fifo_empty),
        .sparse_en        (sparse_en),
        .mode             (mode),
        .bank_switch      (bank_switch),
        .bram_ena         (bram_ena),
        .bram_addr        (bram_addr),
        .bram_din         (bram_din)
    );

    // AXI write task
    task axi_write;
        input [4:0]  addr;
        input [31:0] data;
        begin
            @(posedge clk);
            awaddr  <= addr;  awvalid <= 1;
            wstrb   <= 4'hF;  bready  <= 1;
            @(posedge clk);
            while (!awready) @(posedge clk);
            awvalid <= 0;
            @(posedge clk);
            wdata <= data; wvalid <= 1;
            @(posedge clk);
            while (!wready) @(posedge clk);
            wvalid <= 0;
            while (!bvalid) @(posedge clk);
            bready <= 0;
            @(posedge clk);
        end
    endtask

    // AXI read task
    task axi_read;
        input  [4:0]  addr;
        output [31:0] data;
        begin
            @(posedge clk);
            araddr  <= addr; arvalid <= 1;
            rready  <= 1;
            @(posedge clk);
            while (!arready) @(posedge clk);
            arvalid <= 0;
            while (!rvalid)  @(posedge clk);
            data = rdata;
            rready <= 0;
            @(posedge clk);
        end
    endtask

    integer errors = 0;
    reg [31:0] rd;

    initial begin
        awaddr=0; awvalid=0; wdata=0; wstrb=0; wvalid=0; bready=0;
        araddr=0; arvalid=0; rready=0;
        bsw_seen=0; ena_seen=0; ena_addr=0; ena_din=0;

        repeat(4) @(posedge clk);
        aresetn = 1;
        repeat(2) @(posedge clk);

        // ---- TC1: Write CTRL (sparse_en=1, mode=1, bank_switch=1) ----
        $display("TC1: Write CTRL = 0x07");
        axi_write(5'h00, 32'h00000007);
        @(posedge clk);
        if (sparse_en !== 1 || mode !== 1) begin
            $display("  FAIL: sparse_en=%b mode=%b", sparse_en, mode);
            errors = errors + 1;
        end else if (!bsw_seen) begin
            $display("  FAIL: bank_switch pulse never seen");
            errors = errors + 1;
        end else if (bank_switch !== 0) begin
            $display("  FAIL: bank_switch did not auto-clear");
            errors = errors + 1;
        end else
            $display("  PASS: sparse_en/mode set, bank_switch pulsed and cleared");

        // ---- TC2: Read back CTRL (bank_switch reads 0 - pulse) ----
        $display("TC2: Read CTRL");
        axi_read(5'h00, rd);
        if (rd[1:0] !== 2'b11 || rd[2] !== 1'b0) begin
            $display("  FAIL: rdata[2:0]=%b (expect 011)", rd[2:0]);
            errors = errors + 1;
        end else
            $display("  PASS: rdata=0x%08X", rd);

        // ---- TC3: Read STATUS at 0x04 ----
        $display("TC3: Read STATUS (0x04)");
        axi_read(5'h04, rd);
        $display("  STATUS = 0x%08X", rd);
        if (rd[0]     !== 1'b1)    begin $display("  FAIL: valid_out");        errors=errors+1; end
        if (rd[7:1]   !== 7'd42)   begin $display("  FAIL: zero_skip=%0d",  rd[7:1]);   errors=errors+1; end
        if (rd[13:8]  !== 6'd15)   begin $display("  FAIL: sparse_act=%0d", rd[13:8]);  errors=errors+1; end
        if (rd[19:14] !== 6'd7)    begin $display("  FAIL: sparse_wgt=%0d", rd[19:14]); errors=errors+1; end
        if (rd[23:20] !== 4'b1010) begin $display("  FAIL: sat_flags=%b",   rd[23:20]); errors=errors+1; end
        if (rd[24]    !== 1'b1)    begin $display("  FAIL: fifo_empty bit");            errors=errors+1; end
        if (rd[25]    !== 1'b0)    begin $display("  FAIL: fifo_full bit");             errors=errors+1; end
        if (errors == 0) $display("  PASS");

        // ---- TC4: WADDR write + readback ----
        $display("TC4: WADDR=5, read back");
        axi_write(5'h08, 32'd5);
        axi_read (5'h08, rd);
        if (rd[2:0] !== 3'd5) begin
            $display("  FAIL: WADDR readback=%0d", rd[2:0]); errors=errors+1;
        end else
            $display("  PASS");

        // ---- TC5: Full 128-bit word commit: 3x WDATA_LO + WDATA_HI ----
        $display("TC5: 3x WDATA_LO + WDATA_HI -> bram_ena pulse @ addr 5");
        ena_seen = 0;
        axi_write(5'h0C, 32'h11111111);   // lo[31:0]
        axi_write(5'h0C, 32'h22222222);   // lo[63:32]
        axi_write(5'h0C, 32'h33333333);   // lo[95:64]
        axi_write(5'h10, 32'h44444444);   // hi[127:96] -> commit
        repeat(2) @(posedge clk);
        if (!ena_seen) begin
            $display("  FAIL: bram_ena never pulsed"); errors=errors+1;
        end else if (ena_addr !== 3'd5) begin
            $display("  FAIL: bram_addr=%0d (expect 5)", ena_addr); errors=errors+1;
        end else if (ena_din !== 128'h44444444_33333333_22222222_11111111) begin
            $display("  FAIL: bram_din=0x%032X", ena_din); errors=errors+1;
        end else
            $display("  PASS: din=0x%032X", ena_din);

        // ---- TC6: WADDR auto-increment to 6 ----
        $display("TC6: WADDR auto-increment");
        axi_read(5'h08, rd);
        if (rd[2:0] !== 3'd6) begin
            $display("  FAIL: WADDR=%0d (expect 6)", rd[2:0]); errors=errors+1;
        end else
            $display("  PASS");

        // ---- TC7: Clear sparse_en ----
        $display("TC7: Clear sparse_en via CTRL write");
        axi_write(5'h00, 32'h00000002);  // mode=1, sparse_en=0
        @(posedge clk);
        if (sparse_en !== 0) begin
            $display("  FAIL: sparse_en still asserted"); errors=errors+1;
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