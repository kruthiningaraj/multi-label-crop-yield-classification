from skmultilearn.ensemble import RakelD, RakelO
from skmultilearn.problem_transform import LabelPowerset
from sklearn.tree import DecisionTreeClassifier

def train_rakeld(X, y):
    base_lp = LabelPowerset(DecisionTreeClassifier())
    model = RakelD(base_lp, labelset_size=3)
    model.fit(X, y)
    return model

def train_rakelo(X, y):
    base_lp = LabelPowerset(DecisionTreeClassifier())
    model = RakelO(base_lp, labelset_size=3)
    model.fit(X, y)
    return model
