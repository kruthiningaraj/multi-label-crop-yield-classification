from sklearn.metrics import hamming_loss, accuracy_score, f1_score

def evaluate_model(model, X, y_true, mlb):
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    hamming = hamming_loss(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='micro')
    print(f"Accuracy: {acc:.4f}, Hamming Loss: {hamming:.4f}, F1 Score: {f1:.4f}")
