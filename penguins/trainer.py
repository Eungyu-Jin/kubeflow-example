import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

import argparse
import dill
import logging


def main(
    criterion: str, 
    n_estimators: int, 
    max_depth: int,
    train_ds_path: str
):
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        level=logging.INFO,
    )    

    # numerical
    nums = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    num_tfm = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    # categorical
    cats = ["island", "sex"]
    cat_tfm = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    # column merge
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_tfm, nums),
            ("cat", cat_tfm, cats),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("rf", RandomForestClassifier(
                    criterion= criterion,
                    n_estimators= n_estimators,
                    max_depth= max_depth,
                    oob_score=True,
                    n_jobs = -1
                ))
        ]
    )

    # training
    train_ds = pd.read_csv(train_ds_path)

    X_train = train_ds.iloc[:,1:]
    y_train = train_ds.iloc[:,0].values.ravel()

    model.fit(X_train, y_train)
    val_score = model['rf'].oob_score_
    logging.info(f"accuracy={val_score}")

    with open('model.pkl', 'wb') as f:
        dill.dump(model, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--criterion', type=str)
    parser.add_argument('--n_estimators', type=int)
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--train_ds_path', type=str)
    args = parser.parse_args()

    model = main(
        criterion=args.criterion,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        train_ds_path=args.train_ds_path,
    )