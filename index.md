---
title: "FP4-SPARSA — FP4 Sparse Precision Accelerator Using a Reconfigurable Systolic Array"
---

An FPGA accelerator for low-precision neural-network inference built around a **4×4 weight-stationary systolic array** and **FP4 E2M1** arithmetic.  
It combines **lane-wise zero-skipping**, **on-chip BRAM weight storage**, an **activation FIFO**, and a lightweight **AXI-Lite control path** for a compact accelerator-style design.

---

## What this project explores

FP4-SPARSA is a personal hardware project focused on making neural-network inference more efficient with:

- **FP4 E2M1 quantisation***
- **Per-lane sparsity detection and zero-skipping***
- **Weight-stationary systolic execution***
- **On-chip buffering for weights and activations***
- **A practical FPGA implementation on Xilinx Kintex-7***

The repository includes both the **Verilog RTL** and a **Python demo** for FP4 quantisation and CIFAR-10 / ResNet-20 experimentation.

---

## Architecture

```text
┌─────────────────────────────────────────────┐
│           fp4_sparsa_4x4 (Top)              │
│                                             │
│  ┌──────────────┐   ┌────────────────────┐  │
│  │  AXI-Lite    │   │   BRAM Weight      │  │
│  │  Control     │   │   Buffer           │  │
│  │  (slave)     │   │   (8×128-bit)      │  │
│  └──────┬───────┘   └────────┬───────────┘  │
│         │                    │              │
│  ┌──────▼────────────────────▼───────────┐  │
│  │         systolic_array (4×4)          │  │
│  │                                       │  │
│  │  FP4 decode (decode_fp4 /             │  │
│  │  build_dec_word) at array boundary    │  │
│  │                                       │  │
│  │  ┌──────────┐   ┌──────────┐          │  │
│  │  │pe_pipelined│ │pe_pipelined│  ...   │  │
│  │  │ (5-stage) │ │ (5-stage) │          │  │
│  │  │           │ │           │          │  │
│  │  │ pre-decoded│ │pre-decoded│          │  │
│  │  │ 4-lane MAC│ │ 4-lane MAC│          │  │
│  │  │ pairwise  │ │ pairwise  │          │  │
│  │  │ adder tree│ │ adder tree│          │  │
│  │  │ sparsity  │ │ sparsity  │          │  │
│  │  │ detector  │ │ detector  │          │  │
│  │  └──────────┘   └──────────┘          │  │
│  └───────────────────────────────────────┘  │
│         ▲                                   │
│  ┌──────┴───────┐                           │
│  │  Activation  │                           │
│  │  Input FIFO  │                           │
│  │ (depth-64)   │                           │
│  └──────────────┘                           │
└─────────────────────────────────────────────┘
```

**Key design choices:**

- **FP4 E2M1 format** — 16 representable values with flush-to-zero for subnormals
- **Pre-decode at array boundary** — activations/weights decoded to 8-bit operands in `systolic_array.v` before reaching PEs
- **4-lane MAC structure** — parallel FP4 multiplies inside each PE
- **Weight-stationary dataflow** — weights stay local while activations stream through the array
- **BRAM weight buffer** — on-chip storage (8×128-bit) for repeated reuse across input batches
- **Activation input FIFO** — depth-64 LUTRAM, synchronous
- **18-bit accumulator** — includes guard bit and saturating output
- **Runtime `sparse_en` control** — enable or disable zero-skipping without changing the bitstream
- **AXI-Lite interface** — simple register-based control path with FIFO status reporting

---

## Technical deep-dive

### FP4 E2M1 format

- 1 sign bit, 2 exponent bits, 1 mantissa bit → 16 representable codes
- Subnormal exponent (`exp == 00`) → flushed to zero (FTZ), avoids subnormal handling in hardware
- Values are scaled ×2 (Q1.1) before entering DSP multipliers for compatibility; descaled ÷4 in software post-accumulation

#### FP4 E2M1 decoder table

| Code (S E1E0 M) | Sign | Exp | Mant | FP4 value | Decoded operand (Q1.1 ×2) |
|---|---|---|---|---|---|
| `0000` | + | `00` | `0` | 0 | 0 |
| `0001` | + | `00` | `1` | 0.5 (subnormal, **FTZ**) | 0 |
| `0010` | + | `01` | `0` | 1.0 | 2 |
| `0011` | + | `01` | `1` | 1.5 | 3 |
| `0100` | + | `10` | `0` | 2.0 | 4 |
| `0101` | + | `10` | `1` | 3.0 | 6 |
| `0110` | + | `11` | `0` | 4.0 | 8 |
| `0111` | + | `11` | `1` | 6.0 | 12 |
| `1000` | − | `00` | `0` | −0 | 0 |
| `1001` | − | `00` | `1` | −0.5 (subnormal, **FTZ**) | 0 |
| `1010` | − | `01` | `0` | −1.0 | −2 |
| `1011` | − | `01` | `1` | −1.5 | −3 |
| `1100` | − | `10` | `0` | −2.0 | −4 |
| `1101` | − | `10` | `1` | −3.0 | −6 |
| `1110` | − | `11` | `0` | −4.0 | −8 |
| `1111` | − | `11` | `1` | −6.0 | −12 |

FTZ rows collapse the subnormal magnitude (0.5) to 0 before the value ever reaches the sparsity detector — this is why decode happens once at the array boundary rather than per-PE.

### Decode path

- `decode_fp4` and `build_dec_word` (in `systolic_array.v`) convert raw 4-bit FP4 codes into 8-bit signed operands
- Decode happens once at the array boundary — activations decoded as they enter from the FIFO, weights decoded as they're loaded from BRAM
- PEs receive pre-decoded 8-bit operands directly; no per-PE decode logic, reducing duplication across the 16 PEs
- FTZ applied during decode: subnormal patterns map to 0 before reaching the sparsity detector

### PE pipeline (5-stage, `pe_pipelined.v`)

| Stage | Operation |
|---|---|
| 1 | Operand register (CE-gated capture of pre-decoded act/weight) |
| 2 | 4-lane parallel multiply (DSP48E1, MREG=1/AREG=1) |
| 3 | Per-lane zero detection (sparsity), lane gating |
| 4 | Pairwise adder tree partial-product reduction |
| 5 | 18-bit accumulate with saturation, `valid_out` shift register (depth 5) |

- 4 parallel FP4 MAC lanes per PE → 64 DSP48E1 total across 4×4 array
- 18-bit accumulator includes 1 guard bit; saturation flags (`sat_flags[3:0]`) exposed per PE
- `sparse_en` gates the zero-detection logic at stage 3 — when disabled, all lanes execute unconditionally

### Sparsity detection

- Operates on decoded 8-bit operands (post-FTZ), not raw FP4 codes
- Per-lane: if either activation or weight operand == 0, that lane's MAC is gated off (clock-enable gating, not just result discard)
- `lane_skip_count` / `zero_skip_count[6:0]` track skip statistics per PE
- 2-stage pipelined adder tree compensates for variable lane counts entering the pairwise adder tree, avoiding a WNS hit from combinational skip logic

### Weight buffer (`weight_bram.v`)

- 8×128-bit BRAM (2× BRAM36), holds full 4×4 weight set (16 × 8-bit pre-decoded weights packed per row)
- Load sequence: write `WADDR`, then 3× `WDATA_LO` (32-bit chunks), then `WDATA_HI` (final 32-bit chunk) — `WDATA_HI` write triggers the commit and auto-increments `WADDR`
- 2-cycle BRAM read latency accounted for in the weight-load latency chain feeding the array
- Ping-pong banking (`bank_switch`) allows next weight set to load while current set is in use

### Activation FIFO (`act_fifo.v`)

- Depth-64, synchronous LUTRAM-based FIFO
- Feeds activations row-by-row into the array's left edge
- `fifo_empty` / `fifo_full` flags surfaced directly in AXI `STATUS` register (bits 24/25)
- Drain timing accounts for `acc_s3` hold behavior to avoid bubble cycles at the array boundary

### AXI-Lite control path (`axi_lite_ctrl.v`, v2)

- 5-bit address space, single-cycle register read/write (no wait states)
- `CTRL` (0x00): bit 0 = start, bit 1 = `sparse_en`
- `STATUS` (0x04): bit 0 = done, bit 24 = `fifo_empty`, bit 25 = `fifo_full`
- Weight-load registers (`WADDR`/`WDATA_LO`/`WDATA_HI`) operate independently of the start/done handshake — weights can be loaded while the array is idle
- No interrupt support; status is polled

### Timing closure notes

- `phys_opt_design=ExploreWithRemap` applied permanently — required to close timing at 350MHz with the 5-stage PE pipeline
- MREG=1, AREG=1 on all 64 DSP48E1 instances — needed after the DP4A (4-lane) upgrade increased fan-in to the multiply stage
- BRAM-to-array output timing required an added staging register in `systolic_array.v` to break a critical path through the decode logic
- Final WNS +0.079ns at 350MHz, 17/17 testbench cases passing

---

## FPGA implementation

| Metric | Value |
|---|---|
| **Target Device** | Xilinx Kintex-7 (xc7k160tfbg676-2) |
| **Clock Frequency** | 350 MHz |
| **WNS** | +0.079 ns |
| **DSP48E1 Usage** | 64 (10.67%) |
| **LUT Usage** | 1,926 (1.89%) |
| **Flip-Flop Usage** | 2,899 (1.42%) |
| **BRAM Usage** | 2 (BRAM36) |
| **Dynamic Power** | 0.058 W |
| **Total Power** | 0.170 W |
| **Throughput** | 44.8 GOPS |
| **Efficiency** | 263.5 GOPS/W (total power basis) |
| **Testbench** | 17/17 self-checking test cases pass |

**Toolflow:** Vivado 2024.2 | `phys_opt_design=ExploreWithRemap` | MREG=1, AREG=1 on all DSP48E1

> **Note:** A v20 optimization (act_dec_out forwarding + CE-gated operand registers) was implemented and functionally verified, but worsened both WNS and power versus v19 and was reverted. v19 remains the confirmed final hardware baseline.

---

## Hardware blocks added over time

### BRAM weight buffer
Weights are stored on-chip in 8×128-bit block RAM instead of being streamed directly into the array.

- CPU loads weights via `WADDR` → 3× `WDATA_LO` writes → `WDATA_HI` commit (auto-increments `WADDR`)
- Reduces external weight traffic
- Makes the accelerator behave more like a real inference engine
- Supports repeated reuse across activations

### Activation FIFO
A dedicated depth-64 synchronous FIFO feeds activations row-by-row into the systolic array.

- Decouples input timing from internal pipeline timing
- Status visible via AXI `STATUS` register (empty/full flags)
- Fits naturally with the array dataflow

### AXI-Lite control interface
A minimal 5-bit-address register interface provides software control.

| Register | Offset | Description |
|---|---|---|
| `CTRL` | `0x00` | Start bit, `sparse_en` |
| `STATUS` | `0x04` | Done flag, `fifo_empty` (bit 24), `fifo_full` (bit 25) |
| `WADDR` | `0x08` | Weight BRAM write address |
| `WDATA_LO` | `0x0C` | Weight data [31:0] |
| `WDATA_HI` | `0x10` | Weight data [63:32] — commits write, auto-increments `WADDR` |

---

## Sparsity-aware zero-skipping

The main idea in FP4-SPARSA is **per-lane zero detection**:

- each PE contains 4 parallel FP4 MAC lanes
- activations and weights are pre-decoded to 8-bit operands at the array boundary (FP4 → FTZ applied)
- if a decoded operand lane is zero, that MAC lane is skipped
- the `sparse_en` signal lets you turn the feature on or off at runtime

### Measured sparsity effects on ResNet-20 / CIFAR-10

| Metric | Value |
|---|---|
| Weight sparsity (FP4 + FTZ) | 25.3% |
| Activation sparsity (ReLU + FP4) | 50.5% |
| **MACs skipped** | **63.0%** |
| **Power saving** | **25.2%** |

---

## Python demo

The Python side is used to explore FP4 quantisation behaviour and compare it with full precision.

### Results

| Mode | Accuracy | Notes |
|---|---|---|
| FP32 baseline | **93.17%** | Full precision |
| FP4 quantised | **91.11%** | E2M1 format, FTZ enabled |
| Accuracy drop | **2.06%** | Small degradation |

### Run the demo

```bash
cd python_demo

# Train ResNet-20 on CIFAR-10
python train.py

# Launch the live demo UI
python ui.py
```

The Tkinter UI shows:

- live CIFAR-10 samples
- FP32 vs FP4 predictions
- sparsity and skip-rate statistics

### Requirements

```text
torch (cu121)
torchvision
numpy
tkinter (built-in)
```

---

## Repository structure

```text
FP4-SPARSA/
├── rtl/
│   ├── fp4_sparsa_4x4.v
│   ├── systolic_array.v      # includes decode_fp4 / build_dec_word
│   ├── pe_pipelined.v
│   ├── axi_lite_ctrl.v
│   ├── weight_bram.v
│   └── act_fifo.v
├── tb/
│   ├── tb_fp4_sparsa_4x4.v
│   ├── tb_axi_lite_ctrl.v
│   └── tb_weight_bram.v
├── constraints/
│   └── fp4_sparsa.xdc
├── python_demo/
│   ├── train.py
│   ├── quantize.py
│   ├── cifar10_loader.py
│   ├── ui.py
│   └── models/
│       ├── resnet20.py
│       └── fp4_inference.py
├── LICENSE
└── README.md
```

---

## What makes it interesting

This project explores the combination of:

- **low-precision FP4 arithmetic**
- **systolic-array execution**
- **lane-wise zero-skipping**
- **on-chip buffering**
- **FPGA-friendly streaming control**

That mix makes it a useful personal project for experimenting with compact accelerator design.

---

## Author

**Vinay Navil C N**  
FPGA / digital hardware / AI accelerator projects

[LinkedIn](https://www.linkedin.com/in/vinaynavil/)  
[GitHub](https://github.com/vinaynavil)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
