## =============================================================
##  fp4_sparsa_4x4.xdc  (v19)
##  Target  : xc7k160tfbg676-2
##  Design  : fp4_sparsa_4x4
##
## had set_case_analysis on the clk pin from an old copy paste, removed
## rst -> s_axi_aresetn false path was wrong (no rst port anymore)
## switching activity ports were still named clk/rst, fixed to match
## ============================================================

## ── Device configuration ─────────────────────────────────────
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

## ── Primary clock: 350 MHz (2.857 ns period) ─────────────────
create_clock -period 2.857 -name axi_clk [get_ports s_axi_aclk]

## ── Cut all I/O timing paths (no real board) ─────────────────
set_false_path -from [get_ports -filter {DIRECTION == IN && NAME != s_axi_aclk}]
set_false_path -to   [get_ports -filter {DIRECTION == OUT}]

## ── Active-low synchronous reset false path ──────────────────
set_false_path -from [get_ports s_axi_aresetn]

## ── DSP48E1 pipeline registers ───────────────────────────────
## MREG=1: registers DSP multiply output (P register).
##   Required to break PCOUT->PCIN cascade path between
##   vertically adjacent DSPs in the PE accumulator chain.
##   Without MREG=1 the 3.001 ns cascade delay violates 350 MHz.
##
## AREG=1: pipelines A-input of the pairwise adder tree DSPs
##   (wt_l1_hi_s3a and wt_l1_lo_s3a registered in Stage 3a).
##   Fixes DPIP-1 DRC warning and removes the LUT->DSP.A
##   combinational path. With MREG+AREG the full Stage2->Stage3a
##   path is absorbed into DSP internal registers.
set_property MREG 1 [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]
set_property AREG 1 [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]

## ── phys_opt_design strategy ─────────────────────────────────
## Set in Vivado: Implementation Settings -> phys_opt_design args
## Value: -directive ExploreWithRemap
## ExploreWithRemap: aggressive physical optimisation including
## LUT remapping to improve placement on critical paths.

## ── Power analysis switching activity ────────────────────────
## Clock: always toggling
set_switching_activity \
    -toggle_rate 1.000 -static_probability 0.500 \
    [get_ports s_axi_aclk]

## Input data ports: low toggle rate (sparse FP4 inference)
set_switching_activity \
    -toggle_rate 0.100 -static_probability 0.150 \
    [get_ports -filter {DIRECTION == IN && NAME != s_axi_aclk && NAME != s_axi_aresetn}]

## Reset: rarely asserted
set_switching_activity \
    -toggle_rate 0.010 -static_probability 0.050 \
    [get_ports s_axi_aresetn]

## Output ports: result valid pulses, low toggle rate
set_switching_activity \
    -toggle_rate 0.100 -static_probability 0.150 \
    [get_ports -filter {DIRECTION == OUT}]

