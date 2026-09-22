"""Runtime utilities."""

import random
import time
import numpy as np
import torch

_SCRIPT_START = time.time()

def elapsed():
    s = time.time() - _SCRIPT_START
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def log(msg):
    print(f"[{elapsed()}] {msg}", flush=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
