import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ---------- splits ----------

def load_splits(path: Path) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    for k in ("train", "val", "test"):
        if k not in splits:
            raise KeyError(f"Missing '{k}' in splits.json")
        if not isinstance(splits[k], list):
            raise ValueError(f"{k} must be a list")

    # 防止数据泄漏
    tr, va, te = set(splits["train"]), set(splits["val"]), set(splits["test"])
    if (tr & va) or (tr & te) or (va & te):
        raise ValueError("Overlap detected in splits.json")

    return splits


# ---------- csv ----------

def read_raw_csv(csv_path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    if text_col not in df.columns or label_col not in df.columns:
        raise KeyError(f"CSV columns must contain {text_col}, {label_col}")

    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)
    df = df.reset_index(drop=True)   # ⚠️ very important
    return df


def split_df(df: pd.DataFrame, splits: Dict[str, List[int]]):
    n = len(df)
    ids = splits["train"] + splits["val"] + splits["test"]
    if min(ids) < 0 or max(ids) >= n:
        raise IndexError("splits index out of range")

    train_df = df.iloc[splits["train"]].reset_index(drop=True)
    val_df   = df.iloc[splits["val"]].reset_index(drop=True)
    test_df  = df.iloc[splits["test"]].reset_index(drop=True)
    return train_df, val_df, test_df


# ---------- labels ----------

def build_label_mapping(df: pd.DataFrame, label_col: str):
    labels = sorted(df[label_col].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def make_label_ids(df: pd.DataFrame, label_col: str, label2id: dict):
    return [label2id[x] for x in df[label_col].tolist()]


# ---------- dataset ----------

class BertTextClsDataset(Dataset):
    def __init__(self, df, tokenizer, text_col, label_ids, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_ids = label_ids
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, self.text_col]
        label = int(self.label_ids[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ---------- loaders ----------

def build_loaders(cfg, tokenizer, split: Optional[str] = None):
    df = read_raw_csv(cfg.raw_csv, cfg.text_col, cfg.label_col)
    splits = load_splits(cfg.splits_json)
    train_df, val_df, test_df = split_df(df, splits)

    label2id, id2label = build_label_mapping(df, cfg.label_col)

    def make_loader(sub_df, shuffle):
        ids = make_label_ids(sub_df, cfg.label_col, label2id)
        ds = BertTextClsDataset(sub_df, tokenizer, cfg.text_col, ids, cfg.max_len)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
        )

    if split is None:
        return (
            make_loader(train_df, True),
            make_loader(val_df, False),
            make_loader(test_df, False),
            label2id,
            id2label,
        )

    split = split.lower()
    if split == "train":
        return make_loader(train_df, True), label2id, id2label
    if split == "val":
        return make_loader(val_df, False), label2id, id2label
    if split == "test":
        return make_loader(test_df, False), label2id, id2label

    raise ValueError("split must be train / val / test")
