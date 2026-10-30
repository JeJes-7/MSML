import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

def main(data_path):
    df = pd.read_csv(data_path)

    X = df.drop("DEATH_EVENT", axis=1)
    y = df["DEATH_EVENT"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "random_state": 42
    }

    with mlflow.start_run():
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Manual Logging
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))

        mlflow.sklearn.log_model(model, artifact_path="model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/data_clean.csv"
    )
    args = parser.parse_args()

    main(args.data_path)
