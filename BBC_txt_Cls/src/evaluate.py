from typing import Optional, Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

def evaluate(
    y_true,
    y_pred,
    *,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out: Dict[str, Any] = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc": None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_score is not None:
        try:
            # 多分类 AUC（OVR + macro）；二分类也能用（y_score 形状合适时）
            out["auc"] = float(
                roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
            )
        except ValueError as e:
            out["auc"] = None
            out["auc_reason"] = str(e)

    return out
