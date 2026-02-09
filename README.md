下面是**整合后的完整 README.md（最终版）**，已经**按你的要求**把
👉 **`BBC_DL/checkpoints/*.pth` 的生成说明紧跟在 `train` 命令后面**，逻辑清楚、TA/读者一眼就懂，可直接整体替换你现在的 README。

---

# NLPpj1: ML, DL, and BERT for BBC News Text Classification

## Overview

This project presents a **comparative study of text classification methods** on the **BBC News dataset**, covering three major paradigms in Natural Language Processing:

* **Traditional Machine Learning**
  Bag-of-Words (BoW) / TF-IDF with classical classifiers
* **Deep Learning**
  Word2Vec-based neural networks
* **Transformer-based Models**
  BERT fine-tuning

The task is formulated as a **5-class multi-class text classification problem**, aiming to systematically analyze performance differences across different model families.

---

## Project Structure

```text
.
├── BBC_txt_Cls/
│   ├── src/
│   ├── bow/
│   └── tfidf/
│   Traditional machine learning baselines
│   (BoW / TF-IDF + classical classifiers)
│
├── BBC_DL/
│   ├── src/
│   ├── checkpoints/
│   └── outputs/
│   Deep learning models with Word2Vec embeddings
│   (ANN / CNN / RNN / LSTM)
│
├── BBC_Bert/
│   ├── src/
│   ├── checkpoints/
│   ├── outputs/
│   └── bertviz_repo/
│   Transformer-based model using BERT
│   (fine-tuning, evaluation, visualization)
│
├── result/
│   Aggregated evaluation metrics (JSON files)
│
├── data/
│   Dataset files (user-provided, optional)
│
├── requirements.txt
└── README.md
```

---

## Environment Setup

It is recommended to create and activate a clean Python environment before running the experiments.

```bash
conda create -n nlp python=3.10 -y
conda activate nlp
pip install -r requirements.txt
```

---

## How to Run

### 1. Traditional Machine Learning (BoW / TF-IDF)

Run all traditional machine learning baselines, including:

* Logistic Regression
* Linear SVM
* Naive Bayes
* Random Forest

```bash
cd BBC_txt_Cls
python -m src.run_ml
```

**Output examples**

```text
bow/linear_svm_metrics.json
bow/logistic_regression_metrics.json
bow/naive_bayes_metrics.json
bow/random_forest_metrics.json
tfidf/linear_svm_metrics.json
tfidf/logistic_regression_metrics.json
tfidf/naive_bayes_metrics.json
tfidf/random_forest_metrics.json
```

Each JSON file contains:

* Accuracy
* Macro Precision / Recall / F1-score
* AUC
* Confusion Matrix

---

### 2. Deep Learning (Word2Vec + ANN / CNN / RNN / LSTM)

Deep learning models are trained using **Word2Vec embeddings**.

```bash
cd BBC_DL
```

#### Training

```bash
python -m src.train --model ann
```

After training, the **best-performing model checkpoint** is **automatically saved** to:

```text
BBC_DL/checkpoints/best_ann.pth
```

Available model options and corresponding checkpoints:

* `ann`  → `best_ann.pth`
* `cnn`  → `best_cnn.pth`
* `rnn`  → `best_rnn.pth`
* `lstm` → `best_lstm.pth`

> The checkpoint is saved automatically during training based on validation performance.

#### Evaluation

```bash
python -m src.evaluate --model ann
```

Evaluation metrics and confusion matrices are saved as JSON files in:

```text
BBC_DL/outputs/
```

**Example DL results (ANN)**

* Accuracy: **0.9506**
* Macro Precision / Recall / F1: **0.9505 / 0.9511 / 0.9500**
* AUC: **0.9964**

---

### 3. Transformer-Based Model (BERT)

Fine-tune and evaluate a **BERT-based classifier** on the BBC News dataset.

```bash
cd BBC_Bert
```

#### Training

```bash
python -m src.train
```

After training, the **best BERT checkpoint** is automatically saved to:

```text
BBC_Bert/checkpoints/best_bert.pth
```

#### Evaluation

```bash
python -m src.evaluate
```

Main evaluation output:

```text
eval_bert.json
```

**Example BERT results**

* Loss: **0.0673**
* Accuracy: **0.9820**
* Macro Precision / Recall / F1: **0.9823 / 0.9820 / 0.9821**
* AUC: **0.9995**

---

## Evaluation Metrics

All models are evaluated using:

* Accuracy
* Macro-averaged Precision
* Macro-averaged Recall
* Macro-averaged F1-score
* AUC
* Confusion Matrix

---

## Dataset

BBC News Dataset (Kaggle):

[https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive)

> The dataset is **not included** in this repository.
> Please download it manually and place it under `data/`, or modify dataset paths accordingly.

---

## Model Checkpoints (`.pth` Files)

* Model checkpoints are **generated automatically during training**.
* `.pth` files are **not tracked by Git** and are **excluded from GitHub**.
* To reproduce results, users should re-run the training scripts locally.
* Checkpoints are only required for inference or evaluation without retraining.

---

## Reproducibility Notes

* Ensure consistent dataset paths and preprocessing across ML, DL, and BERT modules.
* Random seeds can be fixed in training scripts for reproducibility.
* Datasets and checkpoints are intentionally excluded from version control following standard ML research practices.

