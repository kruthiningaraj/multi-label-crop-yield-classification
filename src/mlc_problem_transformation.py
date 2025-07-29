from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset

def train_br(X, y):
    model = BinaryRelevance(classifier=RandomForestClassifier())
    model.fit(X, y)
    return model

def train_cc(X, y):
    model = ClassifierChain(classifier=RandomForestClassifier())
    model.fit(X, y)
    return model

def train_lp(X, y):
    model = LabelPowerset(classifier=RandomForestClassifier())
    model.fit(X, y)
    return model
