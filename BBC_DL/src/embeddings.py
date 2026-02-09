import numpy as np
import torch


def build_random_embedding(vocab_size: int, embed_dim: int, pad_id: int = 0):
    emb = torch.randn(vocab_size, embed_dim) * 0.02
    emb[pad_id].zero_()
    return emb


def build_word2vec_embedding(tokenized_texts, stoi, embed_dim: int, window: int, epochs: int, min_count: int):
    """
    Optional: requires gensim. If not installed, caller should fallback to random.
    """
    from gensim.models import Word2Vec

    sentences = tokenized_texts
    w2v = Word2Vec(
        sentences=sentences,
        vector_size=embed_dim,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
    )

    vocab_size = len(stoi)
    emb = np.random.normal(0, 0.02, size=(vocab_size, embed_dim)).astype(np.float32)
    emb[stoi["<pad>"]] = 0.0

    for w, idx in stoi.items():
        if w in ("<pad>", "<unk>"):
            continue
        if w in w2v.wv:
            emb[idx] = w2v.wv[w]

    return torch.tensor(emb, dtype=torch.float32)
