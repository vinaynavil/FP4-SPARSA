<meta name="google-site-verification" content="818WzEt_LHBOTPGBvJDV6GWr8nLYf9EhoDYlegvFvAM" />
# FP4-SPARSA: Sparsity-Aware FP4 Systolic Array MAC Accelerator

An FPGA accelerator for low-precision neural-network inference built around a **4×4 weight-stationary systolic array** and **FP4 E2M1** arithmetic.  
It combines **lane-wise zero-skipping**, **on-chip BRAM weight storage**, an **activation FIFO**, and a lightweight **AXI-Lite control path** for a compact accelerator-style design.

---

## What this project explores

FP4-SPARSA is a personal hardware project focused on making neural-network inference more efficient with:

- **FP4 E2M1 quantisation**
- **Per-lane sparsity detection and zero-skipping**
- **Weight-stationary systolic execution**
- **On-chip buffering for weights and activations**
- **A practical FPGA implementation on Xilinx Kintex-7**

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

- **FP4 E2M1 format** — 16 representable values with flush-to-zero for subnormals
- **4-lane MAC structure** — parallel FP4 multiplies inside each PE
- **Weight-stationary dataflow** — weights stay local while activations stream through the array
- **BRAM weight buffer** — on-chip storage for repeated reuse across input batches
- **Activation input FIFO** — smoother data feeding into the array
- **18-bit accumulator** — includes guard bit and saturating output
- **Runtime `sparse_en` control** — enable or disable zero-skipping without changing the bitstream
- **AXI-Lite interface** — simple register-based control path

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
| **BRAM Usage** | 2 |
| **Dynamic Power** | 0.058 W |
| **Total Power** | 0.170 W |
| **Throughput** | 44.8 GOPS |
| **Efficiency** | 263.5 GOPS/W |
| **Testbench** | 17/17 self-checking test cases pass |

**Toolflow:** Vivado 2024.2 | `phys_opt_design=ExploreWithRemap` | MREG=1, AREG=1 on all DSP48E1

---

## Hardware blocks added over time

### BRAM weight buffer
Weights are stored on-chip in block RAM instead of being streamed directly into the array.

- Reduces external weight traffic
- Makes the accelerator behave more like a real inference engine
- Supports repeated reuse across activations

### Activation FIFO
A dedicated FIFO feeds activations row-by-row into the systolic array.

- Decouples input timing from internal pipeline timing
- Makes the streaming boundary cleaner
- Fits naturally with the array dataflow

### AXI-Lite control interface
A minimal register interface provides software control.

| Register | Offset | Description |
|---|---|---|
| `CTRL` | `0x00` | Start bit, `sparse_en` |
| `STATUS` | `0x04` | Done flag |
| `WADDR` | `0x08` | Weight BRAM write address |
| `WDATA_LO` | `0x0C` | Weight data [31:0] |
| `WDATA_HI` | `0x10` | Weight data [63:32] |

---

## Sparsity-aware zero-skipping

The main idea in FP4-SPARSA is **per-lane zero detection**:

- each PE contains 4 parallel FP4 MAC lanes
- if an activation or weight lane is zero, that MAC lane is skipped
- zero detection happens after FP4 decoding and flush-to-zero handling
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
│   ├── systolic_array.v
│   ├── pe_pipelined.v
│   └── fp4_decoder.v
├── tb/
│   └── tb_fp4_sparsa_4x4.v
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
