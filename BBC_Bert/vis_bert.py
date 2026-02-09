import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_DIR = "/home/mywsl/Workspace/NLP/BBC_Bert"
BERTVIZ_REPO = os.path.join(PROJECT_DIR, "bertviz_repo")  # 你的 repo 目录

os.chdir(PROJECT_DIR)
for p in [PROJECT_DIR, BERTVIZ_REPO]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import Config

# 关键：如果是 repo 形式，直接从 bertviz import
from bertviz import head_view  # 依赖 sys.path 里有 bertviz_repo

def _load_state_dict_any(ckpt_path: str):
    """兼容多种保存格式：{"model_state_dict":...} / {"state_dict":...} / 纯 state_dict"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")

def main():
    cfg = Config()
    device = "cuda" if (getattr(cfg, "device", "cpu") == "cuda" and torch.cuda.is_available()) else "cpu"
    print("DEVICE:", device)

    # 你需要保证 Config 里有这两个字段；如果名字不同就改这里
    model_name = getattr(cfg, "model_name", "bert-base-uncased")
    ckpt_path  = getattr(cfg, "best_ckpt", None)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # num_labels：尽量从 cfg 取；否则先给 5（BBC 通常 5 类）
    num_labels = int(getattr(cfg, "num_labels", 5))

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        output_attentions=True,
    )

    # 如果你有 fine-tune ckpt，就加载；没有就用 base bert 也能可视化 attention
    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = _load_state_dict_any(ckpt_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("[CKPT] loaded:", ckpt_path)
        if missing:
            print("[CKPT] missing keys (ok if head differs):", len(missing))
        if unexpected:
            print("[CKPT] unexpected keys:", len(unexpected))
    else:
        print("[CKPT] not found, use pretrained base model only.")

    model.to(device).eval()

    text = "The government announced new economic policies to support small businesses."
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].detach().cpu().tolist())

    # ====== 输出 HTML（关键修复：HTML 对象 -> 字符串）======
    html_obj = head_view(attentions, tokens, html_action="return")
    html_str = getattr(html_obj, "data", None)
    if html_str is None:
        html_str = str(html_obj)  # 兜底

    out_path = os.path.join(PROJECT_DIR, "outputs", "bertviz_head_view.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    print("[OK] Saved:", out_path)
    print("Open (Windows):")
    print("  explorer.exe", out_path)

if __name__ == "__main__":
    main()
