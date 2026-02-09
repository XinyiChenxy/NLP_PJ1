import os
import random
import numpy as np
from pathlib import Path

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
