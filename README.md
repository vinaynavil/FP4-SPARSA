<meta name="google-site-verification" content="818WzEt_LHBOTPGBvJDV6GWr8nLYf9EhoDYlegvFvAM" />
# FP4-SPARSA: Sparsity-Aware FP4 Systolic Array MAC Accelerator

> **MSc Final Project** — Hemagangothri, Hassan University (2026)  
> Implemented in Verilog RTL | Targeted on Xilinx Kintex-7 FPGA | Verified in Vivado 2024.2

---

## Overview

FP4-SPARSA is a hardware accelerator for neural network inference built around a **4×4 weight-stationary systolic array** using the **FP4 E2M1** floating-point format. It features a novel **sparsity-aware zero-skipping** mechanism that detects and skips zero-valued MAC operations at the per-lane level — directly reducing dynamic power consumption without any loss in correctness.

The project includes both the **RTL hardware design** (Verilog) and a **Python software demo** validating FP4 quantisation accuracy on ResNet-20 / CIFAR-10.

---

## Architecture

```
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
│  │  ┌──────────┐   ┌──────────┐          │  │
│  │  │pe_pipelined│ │pe_pipelined│  ...   │  │
│  │  │ (5-stage) │ │ (5-stage) │          │  │
│  │  │           │ │           │          │  │
│  │  │ fp4_decode│ │ fp4_decode│          │  │
│  │  │ 4-lane MAC│ │ 4-lane MAC│          │  │
│  │  │ Wallace   │ │ Wallace   │          │  │
│  │  │ tree add  │ │ tree add  │          │  │
│  │  │ sparsity  │ │ sparsity  │          │  │
│  │  │ detector  │ │ detector  │          │  │
│  │  └──────────┘   └──────────┘          │  │
│  └───────────────────────────────────────┘  │
│         ▲                                   │
│  ┌──────┴───────┐                           │
│  │  Activation  │                           │
│  │  Input FIFO  │                           │
│  └──────────────┘                           │
└─────────────────────────────────────────────┘
```

**Key design choices:**
- **FP4 E2M1 format** — 16 representable values (0, ±1, ±1.5, ±2, ±3, ±4, ±6), Flush-to-Zero for subnormals
- **DP4A-style PE** — 4 parallel FP4 multiplications per PE per cycle via Wallace tree adder
- **Weight-stationary dataflow** — activations flow left-to-right, weights top-to-bottom
- **128-bit two-phase weight loading** — TPU-style pre-decoded weight registers
- **18-bit accumulator** — includes guard bit with saturating output
- **Runtime reconfigurability** — `sparse_en` port enables/disables zero-skipping without bitstream reload
- **BRAM weight buffer** — weights stored on-chip, loaded once, reused across activations
- **Activation input FIFO** — row-by-row dataflow-aware activation feeding
- **AXI-Lite control interface** — standard CPU-to-accelerator register map

---

## FPGA Implementation Results

| Metric | Value |
|---|---|
| **Target Device** | Xilinx Kintex-7 (xc7k160tfbg676-2) |
| **Clock Frequency** | 350 MHz |
| **WNS** | +0.079 ns |
| **DSP48E1 Usage** | 64 (10.67%) |
| **LUT Usage** | 1,926 (1.89%) |
| **Flip-Flop Usage** | 2,899 (1.42%) |
| **BRAM Usage** | 2 |
| **Dynamic Power** | 0.058 W |
| **Total Power** | 0.170 W |
| **Throughput** | 44.8 GOPS |
| **Efficiency** | 263.5 GOPS/W |
| **Testbench** | 17/17 self-checking TCs pass |

**Tool:** Vivado 2024.2 | `phys_opt_design=ExploreWithRemap` | MREG=1, AREG=1 on all DSP48E1

---

## Implemented Upgrades

### 1. BRAM Weight Buffer

On-chip weight storage using Xilinx block RAM, replacing port-driven weight delivery.

- **Configuration:** 8×128-bit BRAM, `ram_style="block"`
- **Benefit:** Weights loaded once and reused across all activation inputs — matches the behaviour of a real inference accelerator
- **Dataflow:** Ping-pong weight buffers allow weight loading and MAC execution to overlap

### 2. Activation Input FIFO

A dedicated FIFO feeds activations into the systolic array row-by-row.

- **Benefit:** Decouples the host data supply rate from the array's internal pipeline timing
- **Dataflow awareness:** Demonstrates explicit streaming dataflow at the hardware boundary
- **Interface:** Standard valid/ready handshake

### 3. AXI-Lite Control Interface

A minimal AXI-Lite slave register map provides CPU-to-accelerator control — the standard interface for Zynq/Kintex SoC integration.

| Register | Offset | Description |
|---|---|---|
| `CTRL` | `0x00` | Start bit, `sparse_en` |
| `STATUS` | `0x08` | Done flag |
| `WADDR` | `0x08` | Weight BRAM write address |
| `WDATA_LO` | `0x0C` | Weight data [31:0] |
| `WDATA_HI` | `0x10` | Weight data [63:32] |

> These three upgrades collectively bring the design from a standalone MAC array to a complete accelerator subsystem with on-chip storage, dataflow-aware input, and a standard control interface.

---

## Sparsity-Aware Zero-Skipping

The core innovation of FP4-SPARSA is **per-lane dynamic zero detection**:

- Each PE has 4 parallel FP4 MAC lanes
- If either operand (activation or weight) in a lane is zero → that lane's multiply-accumulate is **skipped**
- Zero detection happens at the decoded FP4 level after Flush-to-Zero
- `sparse_en` port allows runtime enable/disable

**Results on ResNet-20 / CIFAR-10:**

| Metric | Value |
|---|---|
| Weight sparsity (FP4 + FTZ) | 25.3% |
| Activation sparsity (ReLU + FP4) | 50.5% |
| **MACs skipped** | **63.0%** |
| **Power saving** | **25.2%** |

> 63% MAC skip = combined effect of weight sparsity + activation sparsity amplified by ReLU and FP4 quantisation

---

## Python Demo (ResNet-20 / CIFAR-10)

Validates that FP4 quantisation preserves accuracy and correlates sparsity metrics with hardware.

### Results

| Mode | Accuracy | Notes |
|---|---|---|
| FP32 baseline | **93.17%** | Full precision |
| FP4 quantised | **91.11%** | E2M1 format, FTZ enabled |
| Accuracy drop | **2.06%** | Within acceptable range |

### Running the Demo

```bash
cd python_demo

# Step 1 — Train ResNet-20 on CIFAR-10 (FP32)
python train.py

# Step 2 — Launch live inference UI
python ui.py
```

The Tkinter UI shows:
- Live image cycling from CIFAR-10 test set
- Side-by-side FP32 vs FP4 predictions
- Hardware correlation panel (sparsity stats, MACs skipped, power saving)

### Requirements

```
torch (cu121)
torchvision
numpy
tkinter (built-in)
```

---

## Repository Structure

```
FP4-SPARSA/
├── rtl/
│   ├── fp4_sparsa_4x4.v      # Top-level module (AXI-Lite + BRAM + FIFO)
│   ├── systolic_array.v      # 4×4 systolic array
│   ├── pe_pipelined.v        # PE: 4-lane FP4 MAC + sparsity detector
│   └── fp4_decoder.v         # FP4 E2M1 decoder with FTZ
├── tb/
│   └── tb_fp4_sparsa_4x4.v   # Self-checking testbench (17 TCs)
├── constraints/
│   └── fp4_sparsa.xdc        # XDC timing + physical constraints
├── python_demo/
│   ├── train.py              # ResNet-20 training (200 epochs)
│   ├── quantize.py           # FP4 E2M1 quantisation engine
│   ├── cifar10_loader.py     # CIFAR-10 data loader
│   ├── ui.py                 # Tkinter live inference UI
│   └── models/
│       ├── resnet20.py       # ResNet-20 architecture
│       └── fp4_inference.py  # FP4 inference engine
├── LICENSE                   # MIT License
└── README.md
```

---

## Novel Contribution

> **Lane-wise FP4 zero-skipping in a systolic array execution unit on FPGA** — no published work combines per-lane dynamic zero-skipping with FP4/sub-byte formats in a systolic array implementation on FPGA.

Related prior work:
- SAVE (Gong 2021) — lane-wise sparsity on CPU in FP32/BF16
- NVIDIA QAD / NVFP4 (Xin et al. 2026) — FP4 on GPU
- FP4-SPARSA — applies lane-wise zero-skipping in FP4 E2M1 in a systolic array on FPGA

---

## Author

**Vinay Navil C N**  
M.Sc. Electronics — Hemagangothri, Hassan University (2026)  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-vinaynavil-blue)](https://www.linkedin.com/in/vinaynavil/)
[![GitHub](https://img.shields.io/badge/GitHub-vinaynavil-black)](https://github.com/vinaynavil)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.