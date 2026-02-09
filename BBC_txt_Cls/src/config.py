from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    project_root: Path = Path(__file__).resolve().parents[2]

    # data
    data_dir: Path = project_root / "data"
    splits_dir: Path = data_dir / "splits"
    dataset_csv: Path = data_dir / "bbc_text_cls.csv"

    # output
    result_dir: Path = project_root / "result"
    reports_dir: Path = result_dir / "reports"

    # reproducibility
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.1
