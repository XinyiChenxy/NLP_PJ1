import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

from src.config import Config
from src.data import build_loaders

logging.set_verbosity_error()


def eval_on_loader(model, loader, device, num_classes: int):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")  # sum then /N

    all_y_true = []
    all_y_pred = []
    all_probs = []
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=ids, attention_mask=mask)
            logits = out.logits

            loss = loss_fn(logits, labels)
            bs = labels.size(0)
            total_loss += float(loss.item())
            total += bs

            probs = torch.softmax(logits, dim=-1)

            preds = torch.argmax(logits, dim=-1)

            all_y_true.append(labels.detach().cpu().numpy())
            all_y_pred.append(preds.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    probs = np.concatenate(all_probs, axis=0)  # shape: [N, C]

    # basic
    acc = float((y_true == y_pred).mean())
    avg_loss = float(total_loss / max(total, 1))

    # macro P/R/F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # AUC (macro)
    # For multiclass AUC, sklearn expects y_true one-hot or labels + probas, and multi_class specified.
    auc_value = None
    auc_note = None
    try:
        if num_classes == 2:
            # binary: use positive class probability
            auc_value = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            # multiclass: one-vs-rest macro
            y_true_oh = np.eye(num_classes, dtype=np.int32)[y_true]  # [N, C]
            auc_value = float(
                roc_auc_score(y_true_oh, probs, average="macro", multi_class="ovr")
            )
    except Exception as e:
        auc_value = None
        auc_note = f"AUC not computed: {type(e).__name__}: {e}"

    metrics = {
        "loss": avg_loss,
        "acc": acc,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "auc": auc_value,
        "confusion_matrix": cm.tolist(),
    }
    if auc_note is not None:
        metrics["auc_note"] = auc_note

    return metrics


def main():
    cfg = Config()
    cfg.ensure_dirs()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    test_loader, label2id, id2label = build_loaders(cfg, tokenizer, split="test")

    num_classes = len(label2id)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_classes,
        label2id=label2id,
        id2label=id2label,
    ).to(cfg.device)

    assert cfg.best_ckpt.exists(), f"Checkpoint not found: {cfg.best_ckpt}"
    state = torch.load(cfg.best_ckpt, map_location=cfg.device)
    model.load_state_dict(state)

    metrics = eval_on_loader(model, test_loader, cfg.device, num_classes=num_classes)
    print(metrics)

    with open(cfg.eval_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] {cfg.eval_path}")


if __name__ == "__main__":
    main()
