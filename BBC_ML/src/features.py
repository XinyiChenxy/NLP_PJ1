from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def build_vectorizer(kind: str, *, max_features=50000, ngram_max=2):
    ngram_range = (1, ngram_max)
    if kind == "bow":
        return CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    if kind == "tfidf":
        return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    raise ValueError("Unknown feature type")
