from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class Config:
    # project layout:
    # NLP/
    #   BBC_DL/   <- project_root
    #   data/     <- shared data folder (bbc_text_cls.csv, splits/)
    project_root: Path = Path(__file__).resolve().parents[1]

    # data lives in ../data
    data_dir: Path = project_root.parent / "data"
    dataset_csv: Path = data_dir / "bbc_text_cls.csv"
    splits_dir: Path = data_dir / "splits"
    splits_json: Path = splits_dir / "splits.json"

    # outputs inside BBC_DL
    ckpt_dir: Path = project_root / "checkpoints"
    out_dir: Path = project_root / "outputs"

    # reproducibility
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.1

    # columns
    text_col: str = "text"
    label_col: str = "labels"

    # ===== EDA-driven length =====
    pct_cap: int = 95
    hard_cap: int = 500

    # ===== Long-tail filtering (EDA) =====
    min_freq: int = 2

    # ===== Training =====
    model: str = "cnn"  # ann/cnn/rnn/lstm
    embed_dim: int = 100
    dropout: float = 0.3
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 30
    patience: int = 4
    num_workers: int = 2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_dirs(self):
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def best_ckpt(self) -> Path:
        return self.ckpt_dir / f"best_{self.model}.pth"

    @property
    def metrics_path(self) -> Path:
        return self.out_dir / f"metrics_{self.model}.json"

    @property
    def eval_path(self) -> Path:
        return self.out_dir / f"eval_{self.model}.json"
