import json
from pathlib import Path

from .config import Config
from .utils import set_seed, ensure_dir
from .dataio import load_dataset, make_or_load_splits, build_splits
from .clean import batch_clean
from .features import build_vectorizer
from .train_ml import train_all_models
from .evaluate import evaluate

def main():
    cfg = Config()
    ensure_dir(cfg.reports_dir)
    ensure_dir(cfg.splits_dir)
    set_seed(cfg.seed)

    df = load_dataset(str(cfg.dataset_csv))
    train_idx, test_idx = make_or_load_splits(
        df, cfg.splits_dir, seed=cfg.seed, test_size=cfg.test_size
    )
    splits = build_splits(df, train_idx, test_idx)

    X_train = batch_clean(splits.X_train)
    X_test = batch_clean(splits.X_test)

    for feat in ["bow", "tfidf"]:
        feat_dir = cfg.reports_dir / feat
        ensure_dir(feat_dir)

        vectorizer = build_vectorizer(feat)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        models = train_all_models(X_train_vec, splits.y_train, seed=cfg.seed)

        for name, model in models.items():
            y_pred = model.predict(X_test_vec)

            y_score = None
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test_vec)
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test_vec)

            metrics = evaluate(splits.y_test, y_pred, y_score=y_score)

            out = feat_dir / f"{name}_metrics.json"
            out.write_text(json.dumps(metrics, indent=2))
            print(f"[{feat}] {name}: {metrics}")

    print("\nDone. Metrics saved to result/reports/{bow,tfidf}/")

if __name__ == "__main__":
    main()


# (nlp) mywsl@LAPTOP-B024L6CK:~/Workspace/NLP/BBC_txt_Cls$ python -m src.run_ml