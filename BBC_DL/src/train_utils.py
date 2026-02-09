import os
import random
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total = 0.0, 0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(ids, mask)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs

    return total_loss / max(total, 1)


@torch.no_grad()
def eval_loss_acc(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        logits = model(ids, mask)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)

        correct += (pred == y).sum().item()
        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs

    return total_loss / max(total, 1), correct / max(total, 1)
