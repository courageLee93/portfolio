from .model_utils import *
import numpy as np
from pathlib import Path
import optuna
import gc
import tensorflow as tf
from tensorflow.keras import metrics
import json
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint



def log_trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    rec = {
        "number": trial.number,
        "value": float(trial.value) if trial.value is not None else None,
        "params": trial.params,
        "user_attrs": dict(trial.user_attrs),
        "state": str(trial.state),
    }
    path = study.user_attrs.get("log_path")
    if path is not None:
        with open(path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def optimize_model_numpy(
        X_train: np.ndarray, X_val: np.ndarray,
        y_train: np.ndarray, y_val: np.ndarray,
        model_name:str,
        save_path: Path,
        class_weight: dict,
        batch_size: int, epoch_num: int, n_trials: int, n_jobs: int = 1
):
    save_path.mkdir(parents=True, exist_ok=True)
    # <editor-fold desc="feature summary">
    seq_len = 376
    feature_len = 11
    feature_binary_idx = [0]  # click
    feature_category_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # campaign ~ cat9
    feature_category_sizes = [675, 9, 70, 1829, 21, 51, 30, 57196, 11, 30]
    # </editor-fold>
    def objective(trial: optuna.trial.Trial):
        # <editor-fold desc="hyper parameters">
        if model_name == "gru":
            if batch_size == 512: lr_min, lr_max = 3e-4, 1.5e-3 # (0.0003 ~ 0.0015)
            else:  lr_min, lr_max = 5e-4, 2e-3 # (0.0005 ~ 0.002)
            hp = {
                # 구조
                "emb_dim": trial.suggest_categorical("emb_dim", [16, 32, 48]),
                "gru_units": trial.suggest_categorical("gru_units", [128, 192]),
                "gru_seq_len": trial.suggest_int("gru_seq_len", 1, 2),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.4),
                "att_units": trial.suggest_categorical("att_units", [128, 192]),
                # 학습
                "lr": trial.suggest_float("lr", lr_min, lr_max, log=True),
            }
        else:
            if batch_size == 512:  lr_min, lr_max = 2e-4, 8e-4
            else:  lr_min, lr_max = 3e-4, 1e-3
            hp = {
                "emb_dim": trial.suggest_categorical("emb_dim", [32, 48]),
                "num_heads": trial.suggest_categorical("num_heads", [2,4]),
                "key_dim": trial.suggest_categorical("key_dim", [32, 48]),
                "encoder_n": trial.suggest_int("encoder_n", 1,3),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.3),
                "att_units": trial.suggest_categorical("att_units", [128, 192]),
                # 학습
                "lr": trial.suggest_float("lr", lr_min, lr_max, log=True),
            }
            dense_dim = hp["num_heads"] * hp["key_dim"]
            if dense_dim > 256:
                raise optuna.TrialPruned()
        # </editor-fold>
        # <editor-fold desc="model load">
        if model_name == "gru":
            model = gru_build_and_compile(hp, seq_len, feature_len,
                                          feature_binary_idx, feature_category_idx, feature_category_sizes)
        else:
            model = transformer_build_and_compile(hp, seq_len, feature_len,
                                                  feature_binary_idx, feature_category_idx, feature_category_sizes)
        # </editor-fold>
        # <editor-fold desc="batch generator load">
        train_gen = numpy_batch_generator(
            X_train, y_train,
            batch_size,
            shuffle=True,
            class_weight=class_weight,
        )
        val_gen = numpy_batch_generator(
            X_val, y_val,
            batch_size,
            shuffle=False,
            class_weight=None,
        )
        steps_per_epoch = int(np.ceil(len(y_train) / batch_size))
        validation_steps = int(np.ceil(len(y_val) / batch_size))
        # </editor-fold>
        # <editor-fold desc="train model">
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_PR-AUC", patience=3, mode="max", restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_PR-AUC", factor=0.5, patience=2, mode="max",
                min_lr=1e-6, verbose=0
            ),
        ]
        print("train...")
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=epoch_num,
            callbacks=callbacks,
            verbose=1,
        )
        # </editor-fold>
        # <editor-fold desc="validate model">
        p_val = model.predict(
            val_gen,
            steps=validation_steps,
            verbose=1,
        ).ravel()
        pr = pr_auc(y_val, p_val)
        f1, thr = best_f1_threshold(y_val, p_val)
        trial.set_user_attr("pr_aucs", float(pr))
        trial.set_user_attr("f1s", float(f1))
        trial.set_user_attr("thrs", float(thr))
        # </editor-fold>
        # <editor-fold desc="memory clear">
        tf.keras.backend.clear_session()
        del train_gen, val_gen
        del model
        gc.collect()
        clear_colab_cache()
        # </editor-fold>
        return float(pr)
    # <editor-fold desc="Optuna 실행(튜닝 실행)">
    study = optuna.create_study(direction="maximize", storage=None)
    study.set_user_attr("log_path", str(save_path / "trial_log.jsonl"))
    study.optimize(objective,
                   n_trials=n_trials, n_jobs=n_jobs,
                   show_progress_bar=True, gc_after_trial=True,
                   callbacks=[log_trial_callback])
    # </editor-fold>
    clear_colab_cache()
    return study


def optimize_model_memmap(
        X_mm: np.ndarray, y_mm: np.ndarray,
        train_index: np.ndarray, val_index: np.ndarray,
        model_name:str,
        save_path: Path,
        class_weight: dict,
        batch_size: int, epoch_num: int, n_trials: int, n_jobs: int = 1
):
    save_path.mkdir(parents=True, exist_ok=True)
    # <editor-fold desc="feature summary">
    seq_len = 376
    feature_len = 11
    feature_binary_idx = [0]  # click
    feature_category_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # campaign ~ cat9
    feature_category_sizes = [675, 9, 70, 1829, 21, 51, 30, 57196, 11, 30]
    # </editor-fold>
    def objective(trial: optuna.trial.Trial):
        # <editor-fold desc="hyper parameters">
        if model_name == "gru":
            if batch_size == 512: lr_min, lr_max = 3e-4, 1.5e-3 # (0.0003 ~ 0.0015)
            else:  lr_min, lr_max = 5e-4, 2e-3 # (0.0005 ~ 0.002)
            hp = {
                # 구조
                "emb_dim": trial.suggest_categorical("emb_dim", [16, 32, 48]),
                "gru_units": trial.suggest_categorical("gru_units", [128, 192]),
                "gru_seq_len": trial.suggest_int("gru_seq_len", 1, 2),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.4),
                "att_units": trial.suggest_categorical("att_units", [128, 192]),
                # 학습
                "lr": trial.suggest_float("lr", lr_min, lr_max, log=True),
            }
        else:
            if batch_size == 512:  lr_min, lr_max = 2e-4, 8e-4
            else:  lr_min, lr_max = 3e-4, 1e-3
            hp = {
                "emb_dim": trial.suggest_categorical("emb_dim", [32, 48]),
                "num_heads": trial.suggest_categorical("num_heads", [2,4]),
                "key_dim": trial.suggest_categorical("key_dim", [32, 48]),
                "encoder_n": trial.suggest_int("encoder_n", 1,3),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.3),
                "att_units": trial.suggest_categorical("att_units", [128, 192]),
                # 학습
                "lr": trial.suggest_float("lr", lr_min, lr_max, log=True),
            }
            dense_dim = hp["num_heads"] * hp["key_dim"]
            if dense_dim > 256:
                raise optuna.TrialPruned()
        # </editor-fold>
        # <editor-fold desc="model load">
        if model_name == "gru":
            model = gru_build_and_compile(hp, seq_len, feature_len,
                                          feature_binary_idx, feature_category_idx, feature_category_sizes)
        else:
            model = transformer_build_and_compile(hp, seq_len, feature_len,
                                                  feature_binary_idx, feature_category_idx, feature_category_sizes)
        # </editor-fold>
        # <editor-fold desc="batch generator load">
        train_ds = memmap_batch_generator(X_mm, y_mm, train_index, batch_size)
        val_ds = memmap_batch_generator(X_mm, y_mm, val_index, batch_size)
        # </editor-fold>
        # <editor-fold desc="train model">
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_PR-AUC", patience=3, mode="max", restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_PR-AUC", factor=0.5, patience=2, mode="max",
                min_lr=1e-6, verbose=0
            ),
        ]
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epoch_num,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )
        # </editor-fold>
        # <editor-fold desc="validate model">
        y_list = []
        for _, yb in val_ds:
            y_list.append(yb.numpy())  # 텐서를 넘파이로 변환
        y_val = np.concatenate(y_list, axis=0)  # (N,)
        # 전체 validation 예측
        p_val = model.predict(val_ds, verbose=1).ravel()  # (N,)
        # metric 계산
        pr = pr_auc(y_val, p_val)
        f1, thr = best_f1_threshold(y_val, p_val)
        # 기록: 후에 study.best_trial.user_attrs로 확인
        trial.set_user_attr("pr_aucs", float(pr))
        trial.set_user_attr("f1s", float(f1))
        trial.set_user_attr("thrs", float(thr))
        # </editor-fold>
        # <editor-fold desc="memory clear">
        tf.keras.backend.clear_session()
        del train_ds, val_ds
        del model
        gc.collect()
        clear_colab_cache()
        # </editor-fold>
        return float(pr)
    # <editor-fold desc="Optuna 실행(튜닝 실행)">
    study = optuna.create_study(direction="maximize", storage=None)
    study.set_user_attr("log_path", str(save_path / "trial_log.jsonl"))
    study.optimize(objective,
                   n_trials=n_trials, n_jobs=n_jobs,
                   show_progress_bar=True, gc_after_trial=True,
                   callbacks=[log_trial_callback])
    # </editor-fold>
    clear_colab_cache()
    return study


def train_model_numpy(
        X_train: np.ndarray, y_train: np.ndarray,
        model_name:str, hyper_params:dict,
        save_path: Path,
        class_weight: dict,
        batch_size: int, epoch_num: int
):
    print("new")
    save_path.mkdir(parents=True, exist_ok=True)
    # <editor-fold desc="feature summary">
    seq_len = 376
    feature_len = 11
    feature_binary_idx = [0]  # click
    feature_category_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # campaign ~ cat9
    feature_category_sizes = [675, 9, 70, 1829, 21, 51, 30, 57196, 11, 30]
    # </editor-fold>
    # <editor-fold desc="model load">
    if model_name == "gru":
        model = gru_build_and_compile(hyper_params, seq_len, feature_len,
                                      feature_binary_idx, feature_category_idx, feature_category_sizes)
    else:
        model = transformer_build_and_compile(hyper_params, seq_len, feature_len,
                                              feature_binary_idx, feature_category_idx, feature_category_sizes)
    # </editor-fold>
    # <editor-fold desc="batch generator load">
    train_gen = numpy_batch_generator(
        X_train, y_train,
        batch_size,
        shuffle=True,
        class_weight=class_weight,
    )
    steps_per_epoch = int(np.ceil(len(y_train) / batch_size))
    # </editor-fold>
    # <editor-fold desc="callback define">
    ckpt_best_model = ModelCheckpoint(
        filepath=str(save_path / "best_model.keras"),
        monitor="PR-AUC",          # metric 이름 그대로
        mode="max",
        save_best_only=True,
        save_weights_only=False,   # 전체 모델 저장
        verbose=1,
    )
    ckpt_best_weights = ModelCheckpoint(
        filepath=str(save_path / "best_weights.weights.h5"),
        monitor="PR-AUC",
        mode="max",
        save_best_only=True,
        save_weights_only=True,    # 가중치만 저장
        verbose=1,
    )
    early_stop = EarlyStopping(
        monitor="PR-AUC",
        patience=3,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="PR-AUC",
        factor=0.5,
        patience=2,
        mode="max",
        min_lr=1e-6,
        verbose=1,
    )
    callbacks = [ckpt_best_model, ckpt_best_weights, early_stop, reduce_lr]
    # </editor-fold>
    # <editor-fold desc="train model">
    print("train...")
    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epoch_num,
        callbacks=callbacks,
        verbose=1,
    )
    # </editor-fold>
    # <editor-fold desc="최종 학습 모델 저장 (전체 모델 + 가중치만 둘 다)">
    # 1) 전체 모델(구조+가중치) 저장 → 나중에 바로 load_model 가능
    #model.save(save_path / "final_model.keras")
    # 2) 가중치만 별도로 저장(원하면)
    #model.save_weights(save_path / "final_weights.ckpt")
    # </editor-fold>
    clear_colab_cache()
    return model


def train_model_memmap(
        X_mm: np.ndarray, y_mm: np.ndarray,
        model_name:str, hyper_params:dict,
        save_path: Path,
        class_weight: dict,
        batch_size: int, epoch_num: int
):
    save_path.mkdir(parents=True, exist_ok=True)
    # <editor-fold desc="feature summary">
    seq_len = 376
    feature_len = 11
    feature_binary_idx = [0]  # click
    feature_category_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # campaign ~ cat9
    feature_category_sizes = [675, 9, 70, 1829, 21, 51, 30, 57196, 11, 30]
    # </editor-fold>
    # <editor-fold desc="model load">
    if model_name == "gru":
        model = gru_build_and_compile(hyper_params, seq_len, feature_len,
                                      feature_binary_idx, feature_category_idx, feature_category_sizes)
    else:
        model = transformer_build_and_compile(hyper_params, seq_len, feature_len,
                                              feature_binary_idx, feature_category_idx, feature_category_sizes)
    # </editor-fold>
    # <editor-fold desc="batch generator load">
    N = y_mm.shape[0]
    all_idx = np.arange(N, dtype=np.int64)
    train_gen = memmap_batch_generator(X_mm, y_mm, all_idx, batch_size=batch_size)
    # </editor-fold>

    # <editor-fold desc="train model">
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="PR-AUC", patience=3, mode="max", restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="PR-AUC", factor=0.5, patience=2, mode="max",
            min_lr=1e-6, verbose=0
        ),
    ]
    print("train...")
    model.fit(
        train_gen,
        epochs=epoch_num,
        callbacks=callbacks,
        verbose=1,
    )
    # </editor-fold>
    # <editor-fold desc="최종 학습 모델 저장 (전체 모델 + 가중치만 둘 다)">
    # 1) 전체 모델(구조+가중치) 저장 → 나중에 바로 load_model 가능
    model.save(save_path / "final_model.keras")
    # 2) 가중치만 별도로 저장(원하면)
    model.save_weights(save_path / "final_weights.ckpt")
    # </editor-fold>
    clear_colab_cache()
    return model

