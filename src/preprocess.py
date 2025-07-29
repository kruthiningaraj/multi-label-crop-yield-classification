import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

def load_and_preprocess(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Crop', 'Yield'])

    # Multi-label encode targets
    y = df[['Crop', 'Yield']].apply(lambda x: list(x), axis=1)
    mlb = MultiLabelBinarizer()
    y_encoded = mlb.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, mlb
