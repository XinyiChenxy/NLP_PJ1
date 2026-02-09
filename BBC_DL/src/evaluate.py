import argparse
import json
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

from .config import Config
from .data import build_dataloaders
from .models import ANNModel, CNNModel, RNNModel, LSTMModel


@torch.no_grad()
def eval_metrics(model, loader, device, num_classes):
    model.eval()
    all_y, all_pred, all_prob = [], [], []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        logits = model(ids, mask)
        prob = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)

        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    probs = np.concatenate(all_prob)

    acc = float((y_true == y_pred).mean())
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()

    auc = None
    try:
        y_oh = np.eye(num_classes, dtype=np.int32)[y_true]
        auc = float(roc_auc_score(y_oh, probs, average="macro", multi_class="ovr"))
    except Exception:
        auc = None

    return {
        "acc": acc,
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "auc": auc,
        "confusion_matrix": cm,
    }


def build_model(cfg, vocab_size, num_classes, pad_id):
    m = cfg.model.lower()
    if m == "ann":
        return ANNModel(vocab_size, cfg.embed_dim, num_classes, dropout=cfg.dropout, pad_id=pad_id)
    if m == "cnn":
        return CNNModel(vocab_size, cfg.embed_dim, num_classes, dropout=cfg.dropout, pad_id=pad_id)
    if m == "rnn":
        return RNNModel(vocab_size, cfg.embed_dim, num_classes, dropout=cfg.dropout, pad_id=pad_id)
    if m == "lstm":
        return LSTMModel(vocab_size, cfg.embed_dim, num_classes, dropout=cfg.dropout, pad_id=pad_id)
    raise ValueError("model must be one of: ann/cnn/rnn/lstm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="ann/cnn/rnn/lstm")
    args = parser.parse_args()

    cfg = Config()
    cfg.model = args.model
    cfg.ensure_dirs()

    train_loader, val_loader, test_loader, meta = build_dataloaders(cfg)
    stoi = meta["stoi"]
    pad_id = stoi["<pad>"]
    num_classes = meta["num_classes"]
    vocab_size = len(stoi)

    model = build_model(cfg, vocab_size, num_classes, pad_id).to(cfg.device)

    assert cfg.best_ckpt.exists(), f"Checkpoint not found: {cfg.best_ckpt}"
    state = torch.load(cfg.best_ckpt, map_location=cfg.device)
    model.load_state_dict(state)

    metrics = eval_metrics(model, test_loader, cfg.device, num_classes)
    print(metrics)

    with open(cfg.eval_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] {cfg.eval_path}")


if __name__ == "__main__":
    main()
