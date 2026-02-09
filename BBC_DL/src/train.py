import argparse
import json
import torch
from torch.optim import AdamW

from .config import Config
from .data import build_dataloaders
from .models import ANNModel, CNNModel, RNNModel, LSTMModel
from .trainer import fit
from .train_utils import set_seed


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
    parser.add_argument("--model", type=str, default="cnn", help="ann/cnn/rnn/lstm")
    args = parser.parse_args()

    cfg = Config()
    cfg.model = args.model
    cfg.ensure_dirs()
    set_seed(cfg.seed)

    train_loader, val_loader, test_loader, meta = build_dataloaders(cfg)

    stoi = meta["stoi"]
    pad_id = stoi["<pad>"]
    num_classes = meta["num_classes"]
    vocab_size = len(stoi)

    print(f"[DATA] max_len={meta['max_len']} | vocab_size={vocab_size} | num_classes={num_classes}")
    print(f"[EDA-ALIGN] pct_cap={cfg.pct_cap}, hard_cap={cfg.hard_cap}, min_freq={cfg.min_freq}")

    model = build_model(cfg, vocab_size, num_classes, pad_id).to(cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    history = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        cfg.device,
        epochs=cfg.epochs,
        patience=cfg.patience,
    )

    torch.save(model.state_dict(), cfg.best_ckpt)
    with open(cfg.metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] {cfg.best_ckpt}")
    print(f"[SAVED] {cfg.metrics_path}")
    print("Next: python -m src.evaluate --model", cfg.model)


if __name__ == "__main__":
    main()
