from preprocess import load_and_preprocess
from mlc_problem_transformation import train_br, train_cc, train_lp
from mlc_adapted_algorithms import train_mlkNN, train_brkNNa
from mlc_ensemble import train_rakeld, train_rakelo
from evaluate import evaluate_model

def train_all_models(train_path):
    X, y, mlb = load_and_preprocess(train_path)
    models = {
        'Binary Relevance': train_br(X, y),
        'Classifier Chains': train_cc(X, y),
        'Label Powerset': train_lp(X, y),
        'MLkNN': train_mlkNN(X, y),
        'BRkNN-a': train_brkNNa(X, y),
        'RAkELd': train_rakeld(X, y),
        'RAkELo': train_rakelo(X, y)
    }
    for name, model in models.items():
        print(f"Evaluating {name}...")
        evaluate_model(model, X, y, mlb)

if __name__ == "__main__":
    train_all_models("data/train.csv")
