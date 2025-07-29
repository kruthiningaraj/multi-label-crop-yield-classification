from skmultilearn.adapt import MLkNN, BRkNNaClassifier

def train_mlkNN(X, y):
    model = MLkNN(k=10)
    model.fit(X, y)
    return model

def train_brkNNa(X, y):
    model = BRkNNaClassifier(k=10)
    model.fit(X, y)
    return model
