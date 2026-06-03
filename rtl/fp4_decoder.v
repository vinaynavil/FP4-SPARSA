`timescale 1ns/1ps
// ============================================================
// Module : fp4_decoder
// Project : FP4-SPARSA 4x4 Systolic Array Accelerator
// Description: Decodes FP4 E2M1 format to signed 8-bit fixed-point
//
//  FP4 E2M1 format: [3]=sign  [2:1]=exponent  [0]=mantissa
//  Bias = 1 (for E2M1)
//
//  Hardware change: Flush-to-Zero (FTZ)
//    Subnormal (exp=00, man=1 → 0.5) flushed to zero.
//    Matches Google TPU / Apple ANE behaviour.
//    Implementation: NOR gate on exp bits gates magnitude to 0.
//    Software-side round-to-nearest-even handles quantization
//    accuracy (NVIDIA transformer-engine approach).
//
//  Decoding table (post-FTZ):
//   exp=00, man=0 → 0.0  → 0
//   exp=00, man=1 → 0.5  → 0  (FLUSHED)
//   exp=01, man=0 → 1.0  → 2
//   exp=01, man=1 → 1.5  → 3
//   exp=10, man=0 → 2.0  → 4
//   exp=10, man=1 → 3.0  → 6
//   exp=11, man=0 → 4.0  → 8
//   exp=11, man=1 → 6.0  → 12
//   Negative: same magnitudes with sign bit set
//
//  Output scaled ×2 (Q1.1): 1.0→2, 1.5→3 ... 6.0→12
// ============================================================
module fp4_decoder (
    input  wire [3:0]        fp4_in,
    output reg  signed [7:0] decoded
);
    wire        sign = fp4_in[3];
    wire [1:0]  exp  = fp4_in[2:1];
    wire        man  = fp4_in[0];

    // FTZ: NOR gate on exp bits - subnormal (exp=00) → magnitude 0
    wire is_subnormal = (exp == 2'b00);

    reg [6:0] magnitude;
    always @(*) begin
        if (is_subnormal) begin
            magnitude = 7'd0;
        end else begin
            case ({exp, man})
                3'b010: magnitude = 7'd2;   // 1.0 → ×2 = 2
                3'b011: magnitude = 7'd3;   // 1.5 → ×2 = 3
                3'b100: magnitude = 7'd4;   // 2.0 → ×2 = 4
                3'b101: magnitude = 7'd6;   // 3.0 → ×2 = 6
                3'b110: magnitude = 7'd8;   // 4.0 → ×2 = 8
                3'b111: magnitude = 7'd12;  // 6.0 → ×2 = 12
                default: magnitude = 7'd0;
            endcase
        end
    end

    always @(*) begin
        if (sign)
            decoded = -$signed({1'b0, magnitude});
        else
            decoded =  $signed({1'b0, magnitude});
    end

endmodule