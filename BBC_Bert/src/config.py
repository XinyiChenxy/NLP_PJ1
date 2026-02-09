from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class Config:
    # ===== ABSOLUTE DATA PATHS =====
    raw_csv: Path = Path("/home/mywsl/Workspace/NLP/data/bbc_text_cls.csv")
    splits_json: Path = Path("/home/mywsl/Workspace/NLP/data/splits/splits.json")

    # ===== OUTPUT DIRS (relative to project) =====
    root: Path = Path(__file__).resolve().parents[1]
    ckpt_dir: Path = root / "checkpoints"
    out_dir: Path = root / "outputs"

    best_ckpt: Path = Path("/home/mywsl/Workspace/NLP/BBC_Bert/checkpoints/best_bert.pth")
    metrics_path: Path = out_dir / "metrics_bert.json"
    eval_path: Path = out_dir / "eval_bert.json"

    # ===== MODEL / TRAINING =====
    model_name: str = "bert-base-uncased"
    max_len: int = 256
    batch_size: int = 16
    lr: float = 2e-5
    epochs: int = 20
    num_workers: int = 2

    # ===== CSV COLUMNS =====
    text_col: str = "text"
    label_col: str = "labels"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_dirs(self):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
