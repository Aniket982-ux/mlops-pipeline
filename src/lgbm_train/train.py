import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import logging
import json
import sys
from pathlib import Path as _Path
import mlflow
import mlflow.lightgbm

# logging + metrics
log_dir = _Path('logs')
log_dir.mkdir(exist_ok=True)
metrics_dir = _Path('metrics')
metrics_dir.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)
if not logger.handlers:
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_dir / 'lgbm.log')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
logger.setLevel(logging.INFO)


def smape(y_pred, dataset):
    y_true = dataset.get_label()
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.zeros_like(y_true)
    nz = denominator != 0
    diff[nz] = np.abs(y_true[nz] - y_pred[nz]) / denominator[nz]
    smape_value = np.mean(diff) * 100
    return 'SMAPE', smape_value, False


def run(input_path: Path, out_model: Path, test_size: float = 0.2, num_boost_round: int = 1000):
    mlflow.set_tracking_uri("http://136.111.62.53:5000")
    mlflow.set_experiment("LGBMTraining")
    # Enable autologging but disable model logging so we can log it manually with registration
    mlflow.lightgbm.autolog(log_models=False)
    
    with mlflow.start_run():
        df = pd.read_pickle(input_path)
        X = np.vstack(df['refined_embedding'].values)
        y = df['price'].values

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val)

        params = {'objective': 'regression', 'metric': 'None', 'verbosity': -1, 'boosting_type': 'gbdt'}

        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'valid'],
            feval=smape,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=10)],
        )

        out_model.parent.mkdir(parents=True, exist_ok=True)
        gbm.save_model(str(out_model))
        logger.info("Saved LightGBM model to %s", out_model)
        
        # Log model to MLflow with registration
        mlflow.lightgbm.log_model(gbm, "model", registered_model_name="LGBMModel")

        # compute validation predictions and save smape metric
        y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration or None)
        denominator = (np.abs(y_val) + np.abs(y_pred)) / 2
        nz = denominator != 0
        diff = np.zeros_like(y_val)
        diff[nz] = np.abs(y_val[nz] - y_pred[nz]) / denominator[nz]
        smape_value = float(np.mean(diff) * 100)
        metrics = {'smape': smape_value}
        with open(metrics_dir / 'lgbm_eval.json', 'w') as f:
            json.dump(metrics, f)
        logger.info("Wrote metrics to %s", (metrics_dir / 'lgbm_eval.json'))
        mlflow.log_metric("val_smape", smape_value)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='input', required=True, type=Path)
    p.add_argument('--out-model', required=True, type=Path)
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--num-boost-round', type=int, default=1000)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    try:
        run(args.input, args.out_model, args.test_size, args.num_boost_round)
    except Exception:
        logger.exception("LightGBM training failed")
        sys.exit(1)

