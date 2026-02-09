import json
import torch
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    logging,
)
from tqdm import tqdm

from src.config import Config
from src.data import build_loaders

logging.set_verbosity_error()


def eval_on_loader(model, loader, device):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    total, correct, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, labels)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            preds = out.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += bs

    return {
        "loss": total_loss / total,
        "acc": correct / total,
    }


def train_one_epoch(model, loader, optimizer, device, epoch, epochs):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()

    total_loss, total = 0.0, 0

    # tqdm
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{epochs}",
        leave=True,
        dynamic_ncols=True,
    )

    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=ids, attention_mask=mask)
        loss = loss_fn(out.logits, labels)

        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total += bs

        # tqdm : loss
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {"loss": total_loss / total}


def main():
    cfg = Config()
    cfg.ensure_dirs()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # train + val
    train_loader, val_loader, _, label2id, id2label = build_loaders(cfg, tokenizer)

    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))
    print("train samples:", len(train_loader.dataset))
    print("val samples:", len(val_loader.dataset))
    print("batch_size:", cfg.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    ).to(cfg.device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    best_val_acc = -1.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n========== Epoch {epoch}/{cfg.epochs} ==========")

        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            cfg.device,
            epoch,
            cfg.epochs,
        )

        va = eval_on_loader(model, val_loader, cfg.device)

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch} | "
            f"train_loss={tr['loss']:.4f} | "
            f"val_loss={va['loss']:.4f} | "
            f"val_acc={va['acc']:.4f}"
        )

        # ✅ val_acc -> best(best acc)
        if va["acc"] > best_val_acc:
            best_val_acc = va["acc"]
            torch.save(model.state_dict(), cfg.best_ckpt)
            print(f"✅ Saved best model (val_acc={best_val_acc:.4f})")

    with open(cfg.metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_acc": best_val_acc,
                "history": history,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[SAVED] {cfg.best_ckpt}")
    print(f"[SAVED] {cfg.metrics_path}")


if __name__ == "__main__":
    main()
