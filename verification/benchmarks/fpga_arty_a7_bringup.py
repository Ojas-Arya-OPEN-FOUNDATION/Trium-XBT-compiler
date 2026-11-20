"""
fpga_arty_a7_bringup.py -- Honest Arty A7 FPGA Hardware Bring-Up Probe.

Detects physical Digilent Arty A7 USB-UART devices. Never fabricates silicon
cycle counts or checksums when no physical board is attached.
"""

from __future__ import annotations

import argparse
import glob
import sys
from typing import Any, Dict, Optional


def find_arty_a7_uart_port() -> Optional[str]:
    for pat in ("/dev/tty.usbserial-*", "/dev/ttyUSB*", "/dev/ttyACM*"):
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None


def run_hardware_bringup(require_hardware: bool = False) -> Dict[str, Any]:
    print("=" * 80)
    print("=== Digilent Arty A7 (Artix-7 XC7A35T) Hardware Bring-Up Probe ===")
    print("=" * 80)

    port = find_arty_a7_uart_port()
    if port is None:
        print("  [SKIP] No physical Arty A7 UART device detected on USB bus.")
        print("  [INFO] No silicon cycle counts reported (offline simulation does not qualify FPGA hardware).")
        print("  [INFO] For RTL simulation, run: python3 tools/run_hardware_benchmarks.py")
        return {
            "board": "Digilent Arty A7 (Xilinx Artix-7 XC7A35T-1CSG324C)",
            "mode": "skipped_no_hardware",
            "hardware_connected": False,
            "status": "SKIPPED",
        }

    print(f"  [HARDWARE] Detected USB-UART device at {port}.")
    print("  [INFO] UART telemetry reader requires pyserial and programmed bitstream from hw/fpga_arty_a7_top.sv.")
    return {
        "board": "Digilent Arty A7 (Xilinx Artix-7 XC7A35T-1CSG324C)",
        "mode": "uart_detected_unprogrammed",
        "port": port,
        "hardware_connected": True,
        "status": "SKIPPED",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-hardware", action="store_true", help="Exit 77 when no physical FPGA is attached.")
    args = parser.parse_args()
    res = run_hardware_bringup(require_hardware=args.require_hardware)
    if args.require_hardware and res["status"] == "SKIPPED":
        sys.exit(77)
