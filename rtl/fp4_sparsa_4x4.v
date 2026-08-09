`timescale 1ns / 1ps
// ============================================================
// Module : fp4_sparsa_4x4
// Project : FP4-SPARSA 
// ============================================================
module fp4_sparsa_4x4 (
    input  wire        s_axi_aclk,
    input  wire        s_axi_aresetn,
    input  wire [4:0]  s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output wire        s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [4:0]  s_axi_araddr,
    input  wire        s_axi_arvalid,
    output wire        s_axi_arready,
    output wire [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output wire        s_axi_rvalid,
    input  wire        s_axi_rready,
    input  wire [15:0]  act_row0,
    input  wire [15:0]  act_row1,
    input  wire [15:0]  act_row2,
    input  wire [15:0]  act_row3,
    input  wire         act_fifo_wr_en,
    output wire [17:0]  result_col0,
    output wire [17:0]  result_col1,
    output wire [17:0]  result_col2,
    output wire [17:0]  result_col3,
    output wire [3:0]   sat_flags,
    output wire         valid_out,
    output wire         fifo_full,
    output wire         fifo_empty
);

    wire        sparse_en_w, mode_w, bank_switch_w;
    wire         bram_ena_w;
    wire [2:0]   bram_waddr_w;
    wire [127:0] bram_wdata_w;
    wire [127:0] bram_rdata_w;
    wire [15:0]  fifo_rd_row0, fifo_rd_row1, fifo_rd_row2, fifo_rd_row3;
    wire         fifo_rd_valid;

    axi_lite_ctrl u_axi_ctrl (
        .s_axi_aclk(s_axi_aclk), 
        .s_axi_aresetn(s_axi_aresetn),
        .s_axi_awaddr(s_axi_awaddr), 
        .s_axi_awvalid(s_axi_awvalid), 
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), 
        .s_axi_wstrb(s_axi_wstrb), 
        .s_axi_wvalid(s_axi_wvalid), 
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp), 
        .s_axi_bvalid(s_axi_bvalid), 
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr), 
        .s_axi_arvalid(s_axi_arvalid), 
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata), 
        .s_axi_rresp(s_axi_rresp), 
        .s_axi_rvalid(s_axi_rvalid), 
        .s_axi_rready(s_axi_rready),
        .valid_out(valid_out), 
        .sat_flags(sat_flags),
        .fifo_full(fifo_full), 
        .fifo_empty(fifo_empty),
        .sparse_en(sparse_en_w), 
        .mode(mode_w), 
        .bank_switch(bank_switch_w),
        .bram_ena(bram_ena_w), 
        .bram_addr(bram_waddr_w), 
        .bram_din(bram_wdata_w)
    );

    // ── BRAM weight buffer ────────────────────────────────────
    // FIX: BRAM Port B read address is registered 1 cycle behind
    // the write address. This ensures dout_b captures the data
    // written at cycle N when sampled at cycle N+1, which is when
    // load_lo_r / load_hi_r fire (they are also registered 1 cycle).
    // Sequence:
    //   Cycle N:   bram_ena=1, addr_a=X, write data into mem[X]
    //              addr_b = addr_a_r (previous addr, e.g. X-1) → dout_b = old
    //   Cycle N+1: bram_ena=0
    //              addr_b = X (registered) → dout_b = mem[X] (just written)
    //              load_lo_r / load_hi_r fire → systolic_array latches dout_b ✓
    reg [2:0] bram_raddr_r;
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) bram_raddr_r <= 3'd0;
        else                 bram_raddr_r <= bram_waddr_w;
    end

    reg load_lo_r0, load_hi_r0;  // 1-cycle registered (intermediate)
    reg load_lo_r,  load_hi_r;   // 2-cycle registered (fires when dout_b valid)

    wire bram_enb_w = bram_ena_w | load_lo_r0 | load_hi_r0 | load_lo_r | load_hi_r;

    weight_bram u_weight_bram (
        .clk(s_axi_aclk),
        .ena_a(bram_ena_w), .addr_a(bram_waddr_w), .din_a(bram_wdata_w),
        .enb_b(bram_enb_w), .addr_b(bram_raddr_r), .dout_b(bram_rdata_w)
    );

    // load_lo_r / load_hi_r: 2-cycle delayed to match BRAM read latency.
    // Cycle N:   bram_ena fires, bram_waddr_w=X, write mem[X]
    //            bram_raddr_r = prev_addr (1c lag)
    // Cycle N+1: bram_raddr_r = X, addr_b presented to BRAM
    //            dout_b = mem[prev] (stale)
    // Cycle N+2: dout_b = mem[X] (valid) <- load fires HERE
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            load_lo_r0 <= 1'b0; load_hi_r0 <= 1'b0;
            load_lo_r  <= 1'b0; load_hi_r  <= 1'b0;
        end else begin
            load_lo_r0 <= bram_ena_w && (bram_waddr_w <= 3'd3);
            load_hi_r0 <= bram_ena_w && (bram_waddr_w >= 3'd4);
            load_lo_r  <= load_lo_r0;
            load_hi_r  <= load_hi_r0;
        end
    end

    // ── Activation FIFO ───────────────────────────────────────
    act_fifo #(.DEPTH(64)) u_act_fifo (
        .clk(s_axi_aclk), .rst(~s_axi_aresetn),
        .wr_en(act_fifo_wr_en),
        .wr_row0(act_row0), .wr_row1(act_row1),
        .wr_row2(act_row2), .wr_row3(act_row3),
        .full(fifo_full),
        .rd_en(~fifo_empty),
        .rd_row0(fifo_rd_row0), .rd_row1(fifo_rd_row1),
        .rd_row2(fifo_rd_row2), .rd_row3(fifo_rd_row3),
        .rd_valid(fifo_rd_valid),
        .empty(fifo_empty)
    );

    // ── Systolic array ────────────────────────────────────────
    systolic_array u_array (
        .clk(s_axi_aclk), .rst(~s_axi_aresetn),
        .sparse_en(sparse_en_w), .mode(mode_w),
        .load_weight_lo(load_lo_r), .load_weight_hi(load_hi_r),
        .bank_switch(bank_switch_w), .weight_data(bram_rdata_w),
        .act_row0(fifo_rd_row0), .act_row1(fifo_rd_row1),
        .act_row2(fifo_rd_row2), .act_row3(fifo_rd_row3),
        .valid_in(fifo_rd_valid),
        .result_col0(result_col0), .result_col1(result_col1),
        .result_col2(result_col2), .result_col3(result_col3),
        .sat_flags(sat_flags),
        .valid_out(valid_out)
    );

endmodule