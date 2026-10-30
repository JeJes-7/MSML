import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main(data_path):
    # Load data clean
    df = pd.read_csv(data_path)

    X = df.drop("DEATH_EVENT", axis=1)
    y = df["DEATH_EVENT"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # AUTolog 
    mlflow.autolog()

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/data_clean.csv"
    )
    args = parser.parse_args()

    main(args.data_path)
