<meta name="google-site-verification" content="818WzEt_LHBOTPGBvJDV6GWr8nLYf9EhoDYlegvFvAM" />

# FP4-SPARSA: Sparsity-Aware FP4 Systolic Array MAC Accelerator

An FPGA accelerator for low-precision neural-network inference built around a **4×4 weight-stationary systolic array** and **FP4 E2M1** arithmetic.  
It combines **lane-wise zero-skipping (operand isolation)**, **on-chip dual-port BRAM weight storage with ping-pong buffering**, an **activation FIFO**, and a lightweight **AXI-Lite control path** for a high-frequency, energy-efficient inference engine.

---

## What this project explores

FP4-SPARSA is a hardware-software co-design project focused on making neural-network inference more efficient with:

- **FP4 E2M1 quantization** (with flush-to-zero and ×2 integer scaling)
- **Fine-grained lane-wise operand isolation** (gating zero-valued multiplier inputs)
- **Weight-stationary systolic spatial execution**
- **On-chip BRAM double-buffering for stall-free weight loading**
- **A fully closed FPGA physical realization on Xilinx Kintex-7 at 350.017 MHz**

The repository includes both the **Verilog RTL** and a **Python demo** for FP4 quantization and CIFAR-10 / ResNet-20 experimentation with an interactive Cyber Dark UI.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                fp4_sparsa_4x4 (Top)                     │
│                                                         │
│  ┌─────────────────┐        ┌────────────────────────┐  │
│  │  AXI-Lite       │        │   Weight BRAM          │  │
│  │  Controller     │        │   (2× BRAM36)          │  │
│  │  (axi_lite_ctrl)│        │   (8×128-bit)          │  │
│  └────────┬────────┘        └───────────┬────────────┘  │
│           │                             │               │
│  ┌────────▼─────────────────────────────▼────────────┐  │
│  │            systolic_array (4×4 Grid)              │  │
│  │                                                   │  │
│  │  FP4 decode (decode_fp4) & raw zero-mask check    │  │
│  │  with double-banked weight storage (Bank A / B)   │  │
│  │                                                   │  │
│  │  ┌──────────────┐      ┌──────────────┐           │  │
│  │  │ pe_pipelined │      │ pe_pipelined │   ...     │  │
│  │  │  (4-stage)   │      │  (4-stage)   │           │  │
│  │  │              │      │              │           │  │
│  │  │  Stage 1: Iso│      │  Stage 1: Iso│           │  │
│  │  │  Stage 2: DSP│      │  Stage 2: DSP│           │  │
│  │  │  Stage 3a:Add│      │  Stage 3a:Add│           │  │
│  │  │  Stage 3b:Acc│      │  Stage 3b:Acc│           │  │
│  │  └──────────────┘      └──────────────┘           │  │
│  └───────────────────────────────────────────────────┘  │
│           ▲                                             │
│  ┌────────┴────────┐                                    │
│  │ Activation FIFO │                                    │
│  │  (depth-64)     │                                    │
│  └─────────────────┘                                    │
└─────────────────────────────────────────────────────────┘
```

**Key design choices:**

- **FP4 E2M1 format** — 16 representable values with flush-to-zero (FTZ) for subnormals and $\times 2$ integer scaling
- **4-lane MAC structure** — parallel FP4 multiplies mapped to dedicated DSP48E1 primitives inside each PE
- **Weight-stationary dataflow** — weights stay local while activations stream left-to-right through the array
- **BRAM weight buffer with Ping-Pong switching** — 8×128-bit dual-port BRAM allowing background weight loads without halting computation
- **Activation input FIFO** — 64-entry LUTRAM FIFO operating in continuous pop mode (`~fifo_empty`)
- **18-bit saturating accumulator** — guard bit protection and overflow clipping
- **Stage 1 Zero-Lane Operand Isolation** — forces multiplier inputs to zero during zero cycles (`sparse_en=1`) to eliminate switching activity
- **AXI-Lite interface** — 5-register memory map for control, status, and 128-bit weight word commits

---

## Technical deep-dive

### FP4 E2M1 format

- 1 sign bit, 2 exponent bits, 1 mantissa bit $\rightarrow$ 16 representable values
- Subnormal exponent (`exp == 00`) $\rightarrow$ flushed to zero (FTZ)
- Values are scaled $\times 2$ to keep data paths integer-only for DSP48E1 multipliers

#### FP4 E2M1 decoder table

| Code (S E1E0 M) | Sign | Exp | Mant | FP4 value | Decoded operand (Scaled ×2) |
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

### PE pipeline (4-stage, `pe_pipelined.v`)

| Stage | Name | Operation |
|---|---|---|
| **Stage 1** | Operand Register & Isolation | Registers FP4 activation/weight inputs; evaluates `calc_zero_mask` and clamps operands (`w_iso`, `a_iso`) to 0 if zero detected |
| **Stage 2** | Parallel Multiplication | $4 \times$ DSP48E1 signed $8 \times 8$ multiplications (`use_dsp="yes"`, `MREG=1`, `AREG=1`) |
| **Stage 3a** | Pairwise Adder Tree | Level-1 registered summation of lane products (`prod0+prod1`, `prod2+prod3`) |
| **Stage 3b** | Accumulator & Saturation | 18-bit accumulation with overflow detection and clipping to 18'h1FFFF / 18'h20000 |

- 4 parallel FP4 MAC lanes per PE $\rightarrow$ 64 DSP48E1 primitives total across the 4×4 array
- 18-bit accumulator output aligned with `valid_out` after 19 clock cycles total latency

### Sparsity detection & Operand Isolation

- Precomputed zero-lane check (`calc_zero_mask`) inspects bits 2:0 of raw FP4 codes during Stage 1
- When `sparse_en=1` and a zero is detected in either weight or activation lane, multiplier inputs are forced to zero before Stage 2
- The DSP48E1 multiplies $0 \times 0$ for that lane, dramatically reducing dynamic switching power without altering pipeline latency or cycle counts

### Weight buffer & Ping-Pong logic (`weight_bram.v`)

- 8×128-bit dual-port BRAM36 block
- 128-bit weight word assembly: 3 sequential 32-bit writes to `WDATA_LO` (0x0C) fill bits 95:0; writing to `WDATA_HI` (0x10) supplies bits 127:96, commits the 128-bit word into BRAM, and auto-increments `WADDR`
- BRAM read address registration (`bram_raddr_r`) ensures stable data output at 350.017 MHz
- Double-banked weight storage (`Bank A` / `Bank B`) with 1-cycle auto-clearing `bank_switch` pulse allows seamless, zero-stall weight updates

### Activation FIFO (`act_fifo.v`)

- Depth-64, synchronous LUTRAM-based FIFO
- Continuous pop mode via `.rd_en(~fifo_empty)` feeds activations to all 4 array rows
- Status signals `fifo_empty` and `fifo_full` exposed in AXI `STATUS` register (bits 24/25)

### AXI-Lite control path (`axi_lite_ctrl.v`)

| Register | Address | Description |
|---|---|---|
| `CTRL` | `0x00` | Control register. Bit 0 = `sparse_en`, Bit 1 = `mode` (FP4/INT4), Bit 2 = `bank_switch` (1-cycle pulse) |
| `STATUS` | `0x04` | Status register. Bit 0 = `valid_out`, Bits 23:20 = `sat_flags[3:0]`, Bit 24 = `fifo_empty`, Bit 25 = `fifo_full` |
| `WADDR` | `0x08` | Weight BRAM write address (3-bit, 0 to 7) |
| `WDATA_LO` | `0x0C` | Weight data low half (3 sequential 32-bit writes fill bits 95:0) |
| `WDATA_HI` | `0x10` | Weight data high half (supplies bits 127:96, commits 128-bit word, auto-increments `WADDR`) |

---

## FPGA implementation & Synthesis Results

| Metric | Verified Value |
|---|---|
| **Target Device** | Xilinx Kintex-7 (`xc7k160tfbg676-2`) |
| **Clock Frequency** | **350.017 MHz** ($T_{clk} = 2.857\text{ ns}$) |
| **Worst Negative Slack (WNS)** | **+0.114 ns** (Timing Closed) |
| **Worst Hold Slack (WHS)** | **+0.084 ns** |
| **DSP48E1 Slice Usage** | **64 / 600 (10.67%)** |
| **Slice LUT Footprint** | **1,963 / 101,400 (1.94%)** (1,877 Logic + 86 Memory) |
| **Flip-Flop Storage** | **3,974 / 202,800 (1.96%)** |
| **Block RAM (BRAM36)** | **2 / 325 (0.62%)** |
| **Dynamic Power** | **0.057 W** (57 mW) |
| **Static Power** | **0.112 W** (112 mW) |
| **Total On-Chip Power** | **0.169 W** (169 mW) (Medium confidence, 18% SAIF net coverage) |
| **Peak Throughput** | **44.8 GOPS** |
| **Energy Efficiency** | **265.1 GOPS/W** (Total power basis) |
| **Verification Suite** | **22 / 22 Testcases PASS** in Vivado XSim (`tb_fp4_sparsa_4x4.v`) |

**Toolflow:** Vivado 2024.2 | `phys_opt_design (ExploreWithRemap)` | `MREG=1`, `AREG=1` on all DSP48E1 instances

---

## Measured Sparsity & Accuracy (ResNet-20 / CIFAR-10)

### Algorithmic Sparsity Statistics

| Metric | Value |
|---|---|
| Weight Sparsity (FP4 + FTZ) | **25.3%** |
| Activation Sparsity (ReLU + FP4) | **50.5%** |
| **MAC Operations Skipped** | **63.0%** |
| **Projected Dynamic Power Saving** | **25.2%** |

### Accuracy Comparison

| Mode | CIFAR-10 Accuracy | Notes |
|---|---|---|
| **FP32 Baseline** | **93.17%** | Full precision ResNet-20 |
| **FP4 Quantized** | **91.11%** | E2M1 format with FTZ |
| **Accuracy Drop** | **2.06%** | Minimal accuracy degradation |

---

## Python Demo & Dashboard

Run the interactive Next-Gen Cyber Dark Dashboard to visualize predictions, real-time confusion matrices, and hardware utilization:

```bash
cd python_demo

# Train ResNet-20 on CIFAR-10 (optional)
python train.py

# Launch the Cyber Dark Accelerator Dashboard UI
python ui.py
```

The Tkinter UI features:
- Live CIFAR-10 test vector preview & ground truth verification
- Side-by-side FP32 vs FP4 model prediction & confidence cards
- 10 Kintex-7 physical metric tiles (`350.017 MHz`, `WNS +0.114ns`, `1,963 LUTs`, `0.169W`, `265.1 GOPS/W`)
- Animated progress bars for accuracy and sparsity metrics

---

## Repository structure

```text
FP4-SPARSA/
├── rtl/
│   ├── fp4_sparsa_4x4.v      # Top-level module
│   ├── systolic_array.v      # 4x4 Systolic array with decode & ping-pong logic
│   ├── pe_pipelined.v        # 4-stage Processing Element with operand isolation
│   ├── axi_lite_ctrl.v       # AXI-Lite control interface
│   ├── weight_bram.v         # Dual-port BRAM weight buffer
│   └── act_fifo.v            # Depth-64 LUTRAM Activation FIFO
├── tb/
│   ├── tb_fp4_sparsa_4x4.v   # Main testbench (22/22 testcases PASS)
│   ├── tb_axi_lite_ctrl.v    # AXI controller unit testbench
│   └── tb_weight_bram.v      # Weight BRAM unit testbench
├── constraints/
│   └── fp4_sparsa.xdc        # 350.017 MHz clock & timing constraints
├── python_demo/
│   ├── train.py              # ResNet-20 CIFAR-10 training
│   ├── quantize.py           # FP4 quantization functions
│   ├── cifar10_loader.py     # CIFAR-10 dataset loader
│   ├── ui.py                 # Next-Gen Cyber Dark Dashboard GUI
│   └── models/
│       └── resnet20.py       # ResNet-20 PyTorch model
├── LICENSE
└── README.md
```

---

## Author

**Vinay Navil C N**  
FPGA / Digital Hardware / AI Accelerator Design

[LinkedIn](https://www.linkedin.com/in/vinaynavil/)  
[GitHub](https://github.com/vinaynavil)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
