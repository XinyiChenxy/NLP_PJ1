from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ----------------------------
# Types
# ----------------------------
PathLike = Union[str, Path]


@dataclass
class DataSplits:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder


# ----------------------------
# Small utils (self-contained)
# ----------------------------
def ensure_dir(p: PathLike) -> Path:
    """Create directory if missing and return it as Path."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------
# Core IO
# ----------------------------
def load_dataset(csv_path: PathLike) -> pd.DataFrame:
    """Load dataset from CSV; expects columns: text, labels."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if "text" not in df.columns or "labels" not in df.columns:
        raise ValueError("Expected columns: text, labels")

    df = df[["text", "labels"]].dropna().reset_index(drop=True)

    # optional: enforce types
    df["text"] = df["text"].astype(str)
    df["labels"] = df["labels"].astype(str)

    return df


def make_or_load_splits(
    df: pd.DataFrame,
    splits_dir: PathLike,
    *,
    seed: int,
    test_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create train/test indices (stratified) and cache to splits.json in splits_dir.
    If splits.json exists, load it for reproducibility.
    """
    splits_dir = ensure_dir(splits_dir)
    split_path = splits_dir / "splits.json"

    if split_path.exists():
        obj = json.loads(split_path.read_text(encoding="utf-8"))
        train_idx = np.array(obj["train"], dtype=int)
        test_idx = np.array(obj["test"], dtype=int)
        return train_idx, test_idx

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=df["labels"],
    )

    split_path.write_text(
        json.dumps(
            {"train": train_idx.tolist(), "test": test_idx.tolist()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return train_idx, test_idx


def build_splits(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> DataSplits:
    """
    Encode labels and slice texts/labels by provided indices.
    """
    le = LabelEncoder()
    y = le.fit_transform(df["labels"].astype(str).values)
    X = df["text"].astype(str).values

    return DataSplits(
        X_train=X[train_idx],
        X_test=X[test_idx],
        y_train=y[train_idx],
        y_test=y[test_idx],
        label_encoder=le,
    )


# ----------------------------
# Optional: persist label encoder for inference
# ----------------------------
def save_label_encoder(label_encoder: LabelEncoder, out_path: PathLike) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    classes = label_encoder.classes_.tolist()
    out_path.write_text(
        json.dumps({"classes": classes}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_label_encoder(path: PathLike) -> LabelEncoder:
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))

    le = LabelEncoder()
    # LabelEncoder expects numpy array for classes_
    le.classes_ = np.array(obj["classes"], dtype=object)
    return le


# ----------------------------
# Demo run (safe to run directly)
# ----------------------------
if __name__ == "__main__":
    
    csv_path = "/home/mywsl/Workspace/NLP/data/bbc.csv"
    splits_dir = "/home/mywsl/Workspace/NLP/data/splits"

    df = load_dataset(csv_path)

    train_idx, test_idx = make_or_load_splits(
        df,
        splits_dir=splits_dir,
        seed=42,
        test_size=0.2,
    )

    splits = build_splits(df, train_idx, test_idx)

    print("Train size:", len(splits.X_train))
    print("Test size :", len(splits.X_test))
    print("Num classes:", len(splits.label_encoder.classes_))
    print("Classes:", splits.label_encoder.classes_)

    # 也可以把 encoder 存起来（推理时用同一个 mapping）
    save_label_encoder(splits.label_encoder, Path(splits_dir) / "label_encoder.json")
    print("Saved label encoder to:", Path(splits_dir) / "label_encoder.json")
