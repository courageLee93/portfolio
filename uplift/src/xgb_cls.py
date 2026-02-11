import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import average_precision_score, log_loss
import optuna
import xgboost as xgb
from typing import List, Optional
import json
import joblib


def _scale_pos_weight(y: np.ndarray) -> float:
    # XGBoost 권장: neg/pos
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / max(pos, 1.0)

def train_xgb_cls(data_load_dir: Path, model_save_dir: Path,
                  treatment: bool,
                  outcome: str,
                  n_folds: int,
                  n_trials: int = 50,
                  n_jobs: int = 7,
                  sample_frac: Optional[float] = None):

    data = pd.read_parquet(data_load_dir / "train.parquet")
    if treatment: data = data[data["treatment"] == 1]
    else: data = data[data["treatment"] == 0]
    covar = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"]

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            # 필수
            "eval_metric": "aucpr",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": 5_000,
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            # 과적합
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            # 정규화 및 규제 관련
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "objective": "binary:logistic",
            # 기타 학습/싫머 관련
            "tree_method": "hist",
            "random_state": 42,
            "nthread": 2,
        }

        pr_aucs: List[float] = []
        lls: List[float] = []
        best_iters: List[int] = []

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(skf.split(data, data[outcome])):

            print(f" Fold {fold} loading...")

            # <editor-fold desc="data load">
            train = data.iloc[train_idx]
            val = data.iloc[val_idx]
            # train data sampling
            if sample_frac is not None:
                train_sampling = StratifiedShuffleSplit(n_splits=1, train_size=sample_frac, random_state=42)
                sub_idx, _ = next(train_sampling.split(train,train[outcome]))
                train = train.iloc[sub_idx]
            X_train = train[covar]
            y_train = train[outcome].astype(int)
            X_val = val[covar]
            y_val = val[outcome].astype(int)
            del train, val
            gc.collect()
            # </editor-fold>

            # <editor-fold desc="compute class weight">
            # 클래스 불균형(class imbalance) 문제를 보정하기 위해 양성(1), 음성(0) 비율을 계산
            spw = _scale_pos_weight(y_train)
            # </editor-fold>
            print(f"Fold {fold} fit...")

            # <editor-fold desc="model train">
            # 모델 정의
            model = xgb.XGBClassifier(**params, scale_pos_weight=spw, early_stopping_rounds=100)
            # 모델 학습
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            # </editor-fold>

            # <editor-fold desc="model validation">
            # 검증셋에 대해 양성클래스(라벨1)의 예측 확률을 출력
            proba = model.predict_proba(X_val)[:, 1]
            pr_auc = float(average_precision_score(y_val, proba))
            pr_aucs.append(pr_auc)
            ll = float(log_loss(y_val, proba))
            lls.append(ll)
            print(f"Fold {fold} done | PR-AUC={pr_auc:.4f}, LL={ll:.4f}")
            # 해당 폴드에서 early stopping으로 선택된 최적 부스팅 라운드(=트리 개수)를 기록
            best_it = getattr(model, "best_iteration_", None)
            if best_it is None:
                best_it = getattr(model, "best_iteration", None)
            if best_it is None:
                best_it = params["n_estimators"]
            best_iters.append(int(best_it))
            # 프루닝
            running_mean = float(np.mean(pr_aucs))
            trial.report(running_mean, step=fold)
            if fold >= 1 and trial.should_prune():
                raise optuna.TrialPruned()
            # </editor-fold>

        trial.set_user_attr("median_best_iter", int(np.median(best_iters)))
        return float(np.mean(pr_aucs))  # 목적값 = PR-AUC 평균 (클수록 좋음)

    # <editor-fold desc="Optuna 실행(튜닝 실행)">
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=0
    )
    study = optuna.create_study(direction="maximize", pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True, gc_after_trial=True)
    # </editor-fold>

    # <editor-fold desc="final train">
    print("최종학습 진행")
    best_params = {
        **study.best_params,
        "objective": "binary:logistic",
        "tree_method": "hist",
        "random_state": 42,
        "eval_metric": "aucpr",
        "n_estimators": int(max(800, study.best_trial.user_attrs.get("median_best_iter", 800) * 1.1)),
        "nthread": 2,
    }
    X = data[covar]
    y = data[outcome].astype(int)
    spw = _scale_pos_weight(y)
    final_model = xgb.XGBClassifier(**best_params, scale_pos_weight=spw)
    final_model.fit(X, y, verbose=True)
    # metric report
    pos_rate = float((y == 1).mean())
    proba = final_model.predict_proba(X)[:, 1]
    pr_auc = float(average_precision_score(y, proba))
    ll = float(log_loss(y, proba))
    report = {
        "pos_rate": pos_rate,
        "pr_auc": pr_auc,
        "log-loss": ll,
        "n_estimators_final": best_params["n_estimators"],
    }
    print(f"Final done | PR-AUC={pr_auc:.4f}, LL={ll:.4f}")
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