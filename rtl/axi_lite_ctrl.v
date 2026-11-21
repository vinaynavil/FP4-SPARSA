`timescale 1ns / 1ps
// ============================================================================
// Module      : axi_lite_ctrl
// Project     : FP4-SPARSA (FP4 Sparse Precision Accelerator using a Reconfigurable Systolic Array)
// Version     : 2.0
// ============================================================================
//
// Register map (byte address, word-aligned):
//   0x00  CTRL        [W/R]
//           [0]  sparse_en
//           [1]  mode
//           [2]  bank_switch
//   0x04  STATUS      [R]
//           [0]      valid_out
//           [23:1]   reserved (0)
//           [24]     fifo_empty
//           [25]     fifo_full
//           [31:26]  reserved (0)
//   0x08  WADDR       [W/R]  [2:0] BRAM write address (0-7)
//   0x0C  WDATA_LO    [W]    bits [95:0] of 128-bit weight word
//                            written as three sequential 32-bit writes
//   0x10  WDATA_HI    [W]    bits [127:96] - triggers BRAM write,
//                            then WADDR auto-increments
//   0x14  SPARE       reserved
//
// CPU weight-load sequence (4x4 matrix = 2 words of 128 bits):
//   1. Load rows 0 & 1:
//        Write WADDR    = 0x0                  (addr 0x08)
//        Write WDATA_LO word 0 (bits  31: 0)   (addr 0x0C)
//        Write WDATA_LO word 1 (bits  63:32)   (addr 0x0C)
//        Write WDATA_LO word 2 (bits  95:64)   (addr 0x0C)
//        Write WDATA_HI        (bits 127:96)   (addr 0x10) -> commits to BRAM addr 0
//   2. Load rows 2 & 3:
//        Write WADDR    = 0x4                  (addr 0x08)
//        Write WDATA_LO word 0 (bits  31: 0)   (addr 0x0C)
//        Write WDATA_LO word 1 (bits  63:32)   (addr 0x0C)
//        Write WDATA_LO word 2 (bits  95:64)   (addr 0x0C)
//        Write WDATA_HI        (bits 127:96)   (addr 0x10) -> commits to BRAM addr 4
//
// Implementation Notes:
//   - WDATA_LO accumulates 96 lower bits across three 32-bit writes via wdata_lo_idx.
//   - Writing WDATA_HI commits the assembled 128-bit word to BRAM and auto-increments WADDR.
//   - bank_switch generates a 1-cycle auto-clearing pulse upon write to CTRL[2].
// ============================================================
module axi_lite_ctrl (
    // AXI-Lite global
    input  wire        s_axi_aclk,
    input  wire        s_axi_aresetn,

    // Write address channel
    input  wire [4:0]  s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output reg         s_axi_awready,

    // Write data channel
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output reg         s_axi_wready,

    // Write response channel
    output reg  [1:0]  s_axi_bresp,
    output reg         s_axi_bvalid,
    input  wire        s_axi_bready,

    // Read address channel
    input  wire [4:0]  s_axi_araddr,
    input  wire        s_axi_arvalid,
    output reg         s_axi_arready,

    // Read data channel
    output reg  [31:0] s_axi_rdata,
    output reg  [1:0]  s_axi_rresp,
    output reg         s_axi_rvalid,
    input  wire        s_axi_rready,

    // Status inputs from systolic array
    input  wire        valid_out,

    // Status inputs from activation FIFO
    input  wire        fifo_full,
    input  wire        fifo_empty,

    // Control outputs
    output reg         sparse_en,
    output reg         mode,
    output reg         bank_switch,

    // BRAM write port outputs
    output reg         bram_ena,
    output reg  [2:0]  bram_addr,
    output reg  [127:0] bram_din
);

    // ── Internal registers ───────────────────────────────────
    reg [4:0]  wr_addr_lat;
    reg        wr_pending;

    // 96-bit accumulation buffer for WDATA_LO (3 × 32-bit writes)
    reg [95:0] wdata_lo_reg;
    reg [1:0]  wdata_lo_idx;   // which 32-bit slot is being written (0,1,2)

    reg [2:0]  waddr_reg;

    // ── Write channel (synchronous reset) ────────────────────
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_awready <= 1'b0;
            s_axi_wready  <= 1'b0;
            s_axi_bvalid  <= 1'b0;
            s_axi_bresp   <= 2'b00;
            wr_pending    <= 1'b0;
            wr_addr_lat   <= 5'd0;
            sparse_en     <= 1'b0;
            mode          <= 1'b0;
            bank_switch   <= 1'b0;
            wdata_lo_reg  <= 96'd0;
            wdata_lo_idx  <= 2'd0;
            waddr_reg     <= 3'd0;
            bram_ena      <= 1'b0;
            bram_addr     <= 3'd0;
            bram_din      <= 128'd0;
        end else begin

            // ── Defaults (pulse signals) ─────────────────────
            bram_ena    <= 1'b0;
            bank_switch <= 1'b0;   // auto-clear: bank_switch is a 1-cycle pulse

            // ── Accept write address ─────────────────────────
            if (s_axi_awvalid && !s_axi_awready && !wr_pending) begin
                s_axi_awready <= 1'b1;
                wr_addr_lat   <= s_axi_awaddr;
                wr_pending    <= 1'b1;
            end else begin
                s_axi_awready <= 1'b0;
            end

            // ── Accept write data ────────────────────────────
            if (s_axi_wvalid && !s_axi_wready && wr_pending && !s_axi_bvalid) begin
                s_axi_wready <= 1'b1;
                wr_pending   <= 1'b0;

                case (wr_addr_lat[4:2])

                    3'b000: begin  // 0x00 CTRL
                        if (s_axi_wstrb[0]) begin
                            sparse_en   <= s_axi_wdata[0];
                            mode        <= s_axi_wdata[1];
                            bank_switch <= s_axi_wdata[2];   // 1-cycle pulse
                        end
                        s_axi_bresp <= 2'b00; // OKAY
                    end

                    3'b001: begin  // 0x04 STATUS - read-only, write returns SLVERR
                        s_axi_bresp <= 2'b10; // SLVERR
                    end

                    3'b010: begin  // 0x08 WADDR
                        if (s_axi_wstrb[0]) begin
                            waddr_reg    <= s_axi_wdata[2:0];
                            wdata_lo_idx <= 2'd0;  // reset sub-index on new WADDR
                        end
                        s_axi_bresp <= 2'b00; // OKAY
                    end

                    3'b011: begin  // 0x0C WDATA_LO
                        // Three sequential 32-bit writes accumulate bits [95:0].
                        // wdata_lo_idx tracks which slot: 0→[31:0], 1→[63:32], 2→[95:64].
                        case (wdata_lo_idx)
                            2'd0: begin
                                if (s_axi_wstrb[0]) wdata_lo_reg[ 7: 0] <= s_axi_wdata[ 7: 0];
                                if (s_axi_wstrb[1]) wdata_lo_reg[15: 8] <= s_axi_wdata[15: 8];
                                if (s_axi_wstrb[2]) wdata_lo_reg[23:16] <= s_axi_wdata[23:16];
                                if (s_axi_wstrb[3]) wdata_lo_reg[31:24] <= s_axi_wdata[31:24];
                                wdata_lo_idx <= 2'd1;
                                s_axi_bresp  <= 2'b00; // OKAY
                            end
                            2'd1: begin
                                if (s_axi_wstrb[0]) wdata_lo_reg[39:32] <= s_axi_wdata[ 7: 0];
                                if (s_axi_wstrb[1]) wdata_lo_reg[47:40] <= s_axi_wdata[15: 8];
                                if (s_axi_wstrb[2]) wdata_lo_reg[55:48] <= s_axi_wdata[23:16];
                                if (s_axi_wstrb[3]) wdata_lo_reg[63:56] <= s_axi_wdata[31:24];
                                wdata_lo_idx <= 2'd2;
                                s_axi_bresp  <= 2'b00; // OKAY
                            end
                            2'd2: begin
                                if (s_axi_wstrb[0]) wdata_lo_reg[71:64] <= s_axi_wdata[ 7: 0];
                                if (s_axi_wstrb[1]) wdata_lo_reg[79:72] <= s_axi_wdata[15: 8];
                                if (s_axi_wstrb[2]) wdata_lo_reg[87:80] <= s_axi_wdata[23:16];
                                if (s_axi_wstrb[3]) wdata_lo_reg[95:88] <= s_axi_wdata[31:24];
                                wdata_lo_idx <= 2'd3;  // Advances to overflow slot
                                s_axi_bresp  <= 2'b00; // OKAY
                            end
                            default: begin
                                // Reject excess write before WDATA_HI to preserve assembled buffer
                                s_axi_bresp  <= 2'b10; // SLVERR (buffer full)
                            end
                        endcase
                    end

                    3'b100: begin  // 0x10 WDATA_HI - commits full 128-bit word
                        // Assemble: [127:96]=WDATA_HI (strobe masked), [95:0]=wdata_lo_reg
                        bram_din  <= {
                            s_axi_wstrb[3] ? s_axi_wdata[31:24] : 8'd0,
                            s_axi_wstrb[2] ? s_axi_wdata[23:16] : 8'd0,
                            s_axi_wstrb[1] ? s_axi_wdata[15: 8] : 8'd0,
                            s_axi_wstrb[0] ? s_axi_wdata[ 7: 0] : 8'd0,
                            wdata_lo_reg
                        };
                        bram_addr <= waddr_reg;
                        bram_ena  <= 1'b1;
                        waddr_reg    <= waddr_reg + 3'd1;  // auto-increment
                        wdata_lo_idx <= 2'd0;              // reset for next word
                        s_axi_bresp  <= 2'b00; // OKAY
                    end

                    default: begin // 0x14 SPARE, 0x18, 0x1C (unmapped addresses)
                        s_axi_bresp <= 2'b10; // SLVERR
                    end

                endcase

                s_axi_bvalid <= 1'b1;
            end else begin
                s_axi_wready <= 1'b0;
            end

            // ── Write response handshake ─────────────────────
            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 1'b0;
        end
    end

    // ── Read channel ─────────────────────────────────────────
    wire [31:0] status_reg = {6'd0, fifo_full, fifo_empty, 23'd0, valid_out};
    wire [31:0] ctrl_reg   = {29'd0, bank_switch, mode, sparse_en};

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rresp   <= 2'b00;
            s_axi_rdata   <= 32'd0;
        end else begin
            if (s_axi_arvalid && !s_axi_arready && !s_axi_rvalid) begin
                s_axi_arready <= 1'b1;
                s_axi_rresp   <= 2'b00;
                s_axi_rvalid  <= 1'b1;
                case (s_axi_araddr[4:2])
                    3'b000:  s_axi_rdata <= ctrl_reg;
                    3'b001:  s_axi_rdata <= status_reg;
                    3'b010:  s_axi_rdata <= {29'd0, waddr_reg};
                    default: s_axi_rdata <= 32'd0;
                endcase
            end else begin
                s_axi_arready <= 1'b0;
            end
            if (s_axi_rvalid && s_axi_rready)
                s_axi_rvalid <= 1'b0;
        end
    end

endmodule