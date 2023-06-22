import pandas as pd
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

def main(
        criterion: str, 
        n_estimators: int, 
        max_depth: int,
        train_data_path: str,
        train_target_path: str
        ):
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        level=logging.INFO,
    )

    # iris = load_iris()
    # data = pd.DataFrame(iris["data"], columns=iris["feature_names"])
    # target = pd.DataFrame(iris["target"], columns=["target"])

    train_data = pd.read_csv(train_data_path)
    train_target = pd.read_csv(train_target_path)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(train_data.values, train_target.values, test_size=0.2)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                    n_estimators= n_estimators, #error시 int 변환
                    max_depth= max_depth,
                    criterion= criterion,
                    oob_score= True
                ))
        ]
    )
    clf.fit(X_train, y_train.ravel())
    y_pred = clf.predict(X_val)
    logging.info(f"accuracy={accuracy_score(y_val, y_pred)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--criterion', type=str)
    parser.add_argument('--n_estimators', type=int)
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--train_target_path', type=str)
    args = parser.parse_args()

    model = main(
        criterion=args.criterion,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        train_ds_path=args.train_ds_path,
    )