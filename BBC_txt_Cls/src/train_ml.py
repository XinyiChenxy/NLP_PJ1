from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def train_all_models(X_train, y_train, seed: int):
    models = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=4000, random_state=seed),
        "linear_svm": LinearSVC(max_iter=20000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
    }
    for m in models.values():
        m.fit(X_train, y_train)
    return models
