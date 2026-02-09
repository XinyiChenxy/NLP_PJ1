import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


_WS = re.compile(r"\s+")
# allow apostrophes inside tokens (align with your EDA tokenizer note)
_NON_WORD = re.compile(r"[^a-zA-Z0-9']+")


def clean_text(text: str) -> str:
    if text is None:
        return ""
    t = str(text).lower()
    t = _NON_WORD.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    toks = [w for w in t.split() if w not in ENGLISH_STOP_WORDS and len(w) > 1]
    return " ".join(toks)


def tokenize(text: str) -> List[str]:
    return text.split()


def compute_max_len(tokenized_texts: List[List[str]], pct_cap: int, hard_cap: int) -> int:
    lengths = np.array([len(t) for t in tokenized_texts], dtype=np.int32)
    max_len = int(np.percentile(lengths, pct_cap))
    max_len = min(max_len, int(hard_cap))
    return max(16, max_len)


def build_vocab(tokenized_texts: List[List[str]], min_freq: int = 2):
    from collections import Counter
    cnt = Counter()
    for toks in tokenized_texts:
        cnt.update(toks)

    PAD = "<pad>"
    UNK = "<unk>"

    itos = [PAD, UNK]
    for w, c in cnt.items():
        if c >= min_freq:
            itos.append(w)
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def numericalize(tokens: List[str], stoi: Dict[str, int]) -> List[int]:
    unk = stoi.get("<unk>", 1)
    return [stoi.get(w, unk) for w in tokens]


def pad_truncate(ids: List[int], max_len: int, pad_id: int = 0):
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


@dataclass
class Splits:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _save_splits(path: Path, train_idx, val_idx, test_idx):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()},
            indent=2,
        ),
        encoding="utf-8",
    )


def make_or_load_splits(df: pd.DataFrame, splits_json: Path, seed: int, test_size: float, val_size: float) -> Splits:
    # If splits exist, support both:
    # 1) {"train": [...], "val": [...], "test": [...]}
    # 2) {"train": [...], "test": [...]}  (older)
    if splits_json.exists():
        obj = json.loads(splits_json.read_text(encoding="utf-8"))
        if "train" in obj and "test" in obj and "val" in obj:
            return Splits(
                train_idx=np.array(obj["train"], dtype=int),
                val_idx=np.array(obj["val"], dtype=int),
                test_idx=np.array(obj["test"], dtype=int),
            )
        if "train" in obj and "test" in obj and "val" not in obj:
            train_idx = np.array(obj["train"], dtype=int)
            test_idx = np.array(obj["test"], dtype=int)

            # create val from train
            y = df.loc[train_idx, "labels"].values
            tr_idx, va_idx = train_test_split(
                train_idx,
                test_size=val_size / (1.0 - test_size),
                random_state=seed,
                stratify=y,
            )
            _save_splits(splits_json, tr_idx, va_idx, test_idx)
            return Splits(train_idx=tr_idx, val_idx=va_idx, test_idx=test_idx)

    # create new splits
    idx = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=df["labels"]
    )
    val_rel = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_rel,
        random_state=seed,
        stratify=df.loc[train_val_idx, "labels"],
    )
    _save_splits(splits_json, train_idx, val_idx, test_idx)
    return Splits(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


class TextClsDataset(Dataset):
    def __init__(self, X_ids: List[List[int]], y: np.ndarray):
        self.X_ids = X_ids
        self.y = y

    def __len__(self):
        return len(self.X_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.X_ids[idx], dtype=torch.long),
            "labels": torch.tensor(int(self.y[idx]), dtype=torch.long),
        }


def collate_fn(batch):
    ids = torch.stack([b["input_ids"] for b in batch], dim=0)  # [B, L]
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    mask = (ids != 0).long()
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def build_dataloaders(cfg):
    assert cfg.dataset_csv.exists(), f"CSV not found: {cfg.dataset_csv}"

    df = pd.read_csv(cfg.dataset_csv)[[cfg.text_col, cfg.label_col]].dropna().reset_index(drop=True)
    df[cfg.text_col] = df[cfg.text_col].astype(str)
    df[cfg.label_col] = df[cfg.label_col].astype(str)
    df = df.rename(columns={cfg.text_col: "text", cfg.label_col: "labels"})

    splits = make_or_load_splits(df, cfg.splits_json, cfg.seed, cfg.test_size, cfg.val_size)

    cleaned = [clean_text(t) for t in df["text"].tolist()]
    tokenized = [tokenize(t) for t in cleaned]

    max_len = compute_max_len(tokenized, cfg.pct_cap, cfg.hard_cap)

    le = LabelEncoder()
    y_all = le.fit_transform(df["labels"].values)

    train_tokens = [tokenized[i] for i in splits.train_idx]
    stoi, itos = build_vocab(train_tokens, min_freq=cfg.min_freq)

    def build_X(indices):
        X = []
        for i in indices:
            ids = numericalize(tokenized[i], stoi)
            ids = pad_truncate(ids, max_len=max_len, pad_id=stoi["<pad>"])
            X.append(ids)
        return X

    X_train = build_X(splits.train_idx)
    X_val = build_X(splits.val_idx)
    X_test = build_X(splits.test_idx)

    y_train = y_all[splits.train_idx]
    y_val = y_all[splits.val_idx]
    y_test = y_all[splits.test_idx]

    train_loader = DataLoader(
        TextClsDataset(X_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TextClsDataset(X_val, y_val),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        TextClsDataset(X_test, y_test),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    meta = {
        "max_len": max_len,
        "stoi": stoi,
        "itos": itos,
        "label_encoder": le,
        "num_classes": len(le.classes_),
    }
    return train_loader, val_loader, test_loader, meta
