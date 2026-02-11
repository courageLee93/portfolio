import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
import xgboost as xgb
from typing import Optional
import json
import joblib


def rmse_metric(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def train_xgb_reg(data_load_dir: Path, model_load_dir: Path, model_save_dir: Path,
                  treatment: bool,
                  outcome: str,
                  n_folds: int,
                  n_trials: int = 50,
                  n_jobs: int = 7,
                  sample_frac: Optional[float] = None):

    data = pd.read_parquet(data_load_dir / "train.parquet")
    if treatment:
        data = data[data["treatment"] == 1]
        cls_model = joblib.load(model_load_dir / "control" / "model.pkl")
    else:
        data = data[data["treatment"] == 0]
        cls_model = joblib.load(model_load_dir / "treatment" / "model.pkl")
    covar = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"]

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            # 필수
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": 5_000,
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            # 과적합 관련
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            # 정규화 및 규제 관련
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "objective": "reg:squarederror",
            # 기타 학습/싫머 관련
            "tree_method": "hist",
            "random_state": 42,
            "nthread": 2,
        }
        rmses, maes, best_iters = [], [], []

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(skf.split(data, data[outcome])):
            print(f" Fold {fold} loading...")
            # <editor-fold desc="data load">
            train = data.iloc[train_idx]
            val = data.iloc[val_idx]
            # train data sampling
            if sample_frac is not None:
                train_sampling = StratifiedShuffleSplit(n_splits=1, train_size=sample_frac, random_state=42)
                sub_idx, _ = next(train_sampling.split(train, train[outcome]))
                train = train.iloc[sub_idx]
            X_train = train[covar]
            y_train = train[outcome].astype(float)
            X_val = val[covar]
            y_val = val[outcome].astype(float)
            # pseudo label
            if treatment:
                p_train = y_train - cls_model.predict_proba(X_train)[:, 1]
                p_val = y_val - cls_model.predict_proba(X_val)[:, 1]
            else:
                p_train = cls_model.predict_proba(X_train)[:, 1] - y_train
                p_val = cls_model.predict_proba(X_val)[:, 1] - y_val
            del train, val
            gc.collect()
            # </editor-fold>

            # <editor-fold desc="model train">
            model = xgb.XGBRegressor(**params, early_stopping_rounds=200)
            # 모델 학습
            model.fit(X_train, p_train, eval_set=[(X_val, p_val)], verbose=False)
            # </editor-fold>

            # <editor-fold desc="model validation">
            pred = model.predict(X_val)
            # RMSE 측정
            rmse = rmse_metric(p_val, pred)
            rmses.append(float(rmse))
            # MAE
            mae = mean_absolute_error(p_val, pred)
            maes.append(float(mae))
            print(f"Fold {fold} done | RMSE={rmse:.5f}, MAE={mae:.5f}")
            # 해당 폴드에서 early stopping으로 선택된 최적 부스팅 라운드(=트리 개수)를 기록
            best_it = getattr(model, "best_iteration_", None)
            if best_it is None:
                best_it = getattr(model, "best_iteration", None)
            if best_it is None:
                best_it = params["n_estimators"]
            best_iters.append(int(best_it))
            # 프루닝
            running_mean = float(np.mean(rmses))
            trial.report(running_mean, step=fold)
            if fold >= 1 and trial.should_prune():
                raise optuna.TrialPruned()
            # </editor-fold>
        trial.set_user_attr("median_best_iter", int(np.median(best_iters)))
        return float(np.mean(rmses))

    # <editor-fold desc="Optuna 실행(튜닝 실행)">
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=0
    )
    study = optuna.create_study(direction="minimize", pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True, gc_after_trial=True)
    # </editor-fold>

    # <editor-fold desc="final train">
    print("최종학습 진행")
    best_params = {
        **study.best_params,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_estimators": int(max(800, study.best_trial.user_attrs.get("median_best_iter", 800) * 1.1)),
        "nthread": 2,
    }
    X = data[covar]
    y = data[outcome].astype(float)
    # pseudo label
    if treatment:
        p = y - cls_model.predict_proba(X)[:, 1]
    else:
        p = cls_model.predict_proba(X)[:, 1] - y
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X, p, verbose=True)
    # metric report
    pred = final_model.predict(X)
    rmse = rmse_metric(p, pred)
    mae = mean_absolute_error(p, pred)
    report = {
        "rmse": rmse,
        "mae": mae,
        "n_estimators_final": best_params["n_estimators"],
    }
    print(f"Fold final done | RMSE={rmse:.5f}, MAE={mae:.5f}")
    # save
    model_save_dir.mkdir(parents=True, exist_ok=True)
    # 파라미터 저장
    with open(model_save_dir / f"best_params.json", "w") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    # 모델 저장
    joblib.dump(final_model, model_save_dir / f"model.pkl")
    # 리포트 저장
    with open(model_save_dir / f"report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    # </editor-fold>

    return final_model, best_params, study, report