from typing import Dict
import torch
from .train_utils import train_one_epoch, eval_loss_acc


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    *,
    epochs: int,
    patience: int = 3,
) -> Dict:
    best_val_acc = -1.0
    best_state = None
    wait = 0

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc = eval_loss_acc(model, val_loader, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_loss:.4f} | "
            f"val_loss={va_loss:.4f} | "
            f"val_acc={va_acc:.4f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}. Best val_acc={best_val_acc:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_val_acc"] = best_val_acc
    history["epochs_ran"] = epoch
    return history
