set_property MREG 1 [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]
## ============================================================
##  fp4_sparsa_4x4.xdc  (NO-HARDWARE VERSION)
##  Target  : xc7k160tfbg676-2
##  Design  : fp4_sparsa_4x4
##
##  Notes:
##    - No pin assignments (software-only flow)
##      Vivado auto-places IO; all timing/power reports valid
##    - False paths on all IO cuts setup/hold analysis at ports
##    - MREG=1 on DSP48E1: required to break PCOUT→PCIN cascade
##      path between vertically adjacent DSPs in each PE.
##      Without MREG=1 the 3.001ns cascade delay violates 300MHz.
##    - set_msg_config removed (not supported in XDC files)
##    - CFGBVS/CONFIG_VOLTAGE set for 3.3V single supply board
##    - AVAL-6 warnings are cosmetic only - MREG=1 is correct
##      for this design despite zero-skip mux pass-through mode
##
##  Changes from previous version:
##    - sparse_en port added (covered by DIRECTION==IN filter)
##    - mode port added (covered by DIRECTION==IN filter)
##    - Clock updated to 350MHz (2.857ns period)
##    - Power constraints tightened:
##        Clock net: toggle_rate 1.000 (always toggling, fixed)
##        Input ports: toggle_rate 0.100 (sparse inference traffic)
##        Output ports: toggle_rate 0.100 (result valid pulses)
##        Static probability 0.150 (weights mostly zero in FP4)
##      Rationale: FP4 sparsity means most activations/weights
##      are zero → low switching activity on data ports.
##      Clock always toggles at full rate (1.0 = 100%).
## ============================================================
## ── Device configuration ─────────────────────────────────────
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
## ── Primary clock (350 MHz → 2.857 ns) ──────────────────────
create_clock -period 2.857 -name sys_clk -waveform {0.000 1.429} [get_ports clk]
## ── Cut all I/O timing paths (no real board) ─────────────────
set_false_path -from [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_false_path -to [get_ports -filter {DIRECTION == OUT}]
## ── Async reset false path ────────────────────────────────────
set_false_path -from [get_ports rst]
## ── DSP48E1 MREG=1 (Implementation only) ────────────────────
## ── Switching activity for power analysis ────────────────────
## Clock: always toggling at full rate
set_switching_activity -toggle_rate 1.000 -static_probability 0.500 [get_ports clk]
## Input data ports: low toggle rate (sparse FP4 inference)
## Most activations/weights are zero → low switching
set_switching_activity -toggle_rate 0.100 -static_probability 0.150 [get_ports -filter {DIRECTION == IN && NAME != clk && NAME != rst}]
## Reset: rarely asserted
set_switching_activity -toggle_rate 0.010 -static_probability 0.050 [get_ports rst]
## Output ports: valid pulses + result data, low toggle rate
set_switching_activity -toggle_rate 0.100 -static_probability 0.150 [get_ports -filter {DIRECTION == OUT}]


report_timing -from [get_cells *skip_wire*] -to [get_cells *zero_skip_count*]


