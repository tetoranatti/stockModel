from __future__ import annotations

import compileall
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
ok = compileall.compile_dir(str(root), quiet=1)
if not ok:
    raise SystemExit("syntax compilation failed")
print("syntax compilation: OK")
print("Run project-level import tests from F:\\stockModel\\research where external modules are available.")
