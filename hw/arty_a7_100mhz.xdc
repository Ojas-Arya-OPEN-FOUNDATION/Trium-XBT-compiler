# =============================================================================
# File: hw/arty_a7_100mhz.xdc
# Description: Digilent Arty A7 (XC7A35T / XC7A100T) 100 MHz Static Timing
#              Analysis (STA) & Pinout Constraints for hw/fpga_arty_a7_top.sv.
#              Port names match hw/fpga_arty_a7_top.sv verbatim.
# =============================================================================

## 100 MHz System Clock (Pin E3 on Digilent Arty A7)
set_property -dict { PACKAGE_PIN E3    IOSTANDARD LVCMOS33 } [get_ports { clk_100m }];
create_clock -add -name sys_clk_pin -period 10.000 -waveform {0.000 5.000} [get_ports { clk_100m }];

## Active-Low Reset Push Button (CK_RST on Arty A7, Pin C2)
set_property -dict { PACKAGE_PIN C2    IOSTANDARD LVCMOS33 } [get_ports { btn_rst_n }];

## 4 User Status LEDs (LD4..LD7 on Arty A7)
set_property -dict { PACKAGE_PIN H5    IOSTANDARD LVCMOS33 } [get_ports { led[0] }];
set_property -dict { PACKAGE_PIN J5    IOSTANDARD LVCMOS33 } [get_ports { led[1] }];
set_property -dict { PACKAGE_PIN T9    IOSTANDARD LVCMOS33 } [get_ports { led[2] }];
set_property -dict { PACKAGE_PIN T10   IOSTANDARD LVCMOS33 } [get_ports { led[3] }];

## USB-UART Bridge (FT2232HQ on Arty A7)
set_property -dict { PACKAGE_PIN A9    IOSTANDARD LVCMOS33 } [get_ports { uart_rx }];
set_property -dict { PACKAGE_PIN D10   IOSTANDARD LVCMOS33 } [get_ports { uart_tx }];

## Asynchronous I/O Timing Constraints (2.0 ns margin relative to 100 MHz clock)
set_input_delay  -clock [get_clocks sys_clk_pin] -max 2.000 [get_ports { btn_rst_n uart_rx }];
set_input_delay  -clock [get_clocks sys_clk_pin] -min 0.000 [get_ports { btn_rst_n uart_rx }];
set_output_delay -clock [get_clocks sys_clk_pin] -max 2.000 [get_ports { led[*] uart_tx }];
set_output_delay -clock [get_clocks sys_clk_pin] -min 0.000 [get_ports { led[*] uart_tx }];
set_false_path   -from [get_ports { btn_rst_n }];
