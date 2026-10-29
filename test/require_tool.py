#!/usr/bin/env python3
"""CTest adapter: dependency absence is an explicit skip, never a pass."""
import shutil
import subprocess
import sys
if not shutil.which(sys.argv[1]):
    print(f"SKIP: required tool {sys.argv[1]} is not installed")
    sys.exit(77)
sys.exit(subprocess.call(sys.argv[2:]))
