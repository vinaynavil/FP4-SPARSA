## ============================================================
##  fp4_sparsa.xdc  (v20 Optimized XDC Constraints)
##  Target Device   : xc7k160tfbg676-2 (Kintex-7)
##  Top Module      : fp4_sparsa_4x4
## ============================================================

## ── Device configuration ───────────────────────────────────
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

## ── Primary clock: 350 MHz (2.857 ns period) ─────────────────
create_clock -period 2.857 -name axi_clk [get_ports s_axi_aclk]
set_input_jitter [get_clocks axi_clk] 0.050

## ── Out-of-Context I/O Timing Paths ──────────────────────────
set_false_path -from [get_ports -filter {DIRECTION == IN && NAME != s_axi_aclk}]
set_false_path -to   [get_ports -filter {DIRECTION == OUT}]
set_false_path -from [get_ports s_axi_aresetn]

## ── Multicycle Paths for Static Control Registers ────────────
set_multicycle_path -setup -from [get_cells -hierarchical -filter {NAME =~ *u_axi_ctrl*sparse_en*}] 2
set_multicycle_path -hold  -from [get_cells -hierarchical -filter {NAME =~ *u_axi_ctrl*sparse_en*}] 1
set_multicycle_path -setup -from [get_cells -hierarchical -filter {NAME =~ *u_axi_ctrl*mode*}] 2
set_multicycle_path -hold  -from [get_cells -hierarchical -filter {NAME =~ *u_axi_ctrl*mode*}] 1

## ── DSP48E1 Primitive Hardware Optimization ─────────────────
## AREG/BREG/MREG/PREG = 1: Force full pipeline register absorption into DSP blocks.
## USE_DPORT = FALSE: Disable unused pre-adder hardware to save dynamic power.
set_property AREG 1 [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]
set_property BREG 1 [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]
set_property MREG 1 [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]
set_property PREG 1 [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]
set_property USE_DPORT FALSE [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]

## ── RAMB36E1 Primitive Hardware Optimization ─────────────────
## DOA_REG / DOB_REG = 1: Enable internal BRAM output registers for optimal timing.
set_property DOA_REG 1 [get_cells -hierarchical -filter {REF_NAME == RAMB36E1}]
set_property DOB_REG 1 [get_cells -hierarchical -filter {REF_NAME == RAMB36E1}]

## ── Reset Switching Activity (fixes Power 33-332 warning) ────
set_switching_activity \
    -toggle_rate 0.100 -static_probability 0.999 \
    [get_ports s_axi_aresetn]
