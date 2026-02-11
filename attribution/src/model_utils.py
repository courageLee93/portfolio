import numpy as np
from pathlib import Path
import gc
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score
import tensorflow as tf
from tensorflow.keras import metrics, optimizers, losses
from .model_defs import gru_att_cls, transformer_att_cls
import shutil
import os


def clear_colab_cache():
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass

    gc.collect()

    # 로컬에 쌓이는 로그/캐시 폴더 일부 삭제
    targets = [
        "/content/tmp",
        "/content/.config/logs",
        "/content/.local/share/optuna",
        "/root/.local/share/optuna",
    ]
    for p in targets:
        try:
            shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass
        if p.endswith("tmp"):
            os.makedirs(p, exist_ok=True)


def compute_class_weight(data_path: Path):
    y = np.load(data_path / "y.npy", mmap_mode="r")
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    class_weight = {0: 1.0, 1: round(n_neg / n_pos)}
    return class_weight


def memmap_batch_generator(X_mm, y_mm, indices, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(indices)
    def _load_from_memmap(i):
        i = int(i)
        x = np.asarray(X_mm[i], dtype=np.float32)
        y = np.asarray(y_mm[i], dtype=np.int8)
        return x, y
    def _tf_map(i): # i --> x[i], y[i]
        x, y = tf.numpy_function(_load_from_memmap, [i], [tf.float32, tf.int8])
        x.set_shape((X_mm.shape[1], X_mm.shape[2]))  # (T, F) # X[i] → (seq_len, feature_len)
        y.set_shape(()) # y[i] → ()(스칼라 라벨)
        return x, y
    ds = ds.map(_tf_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def numpy_batch_generator(X, y, batch_size, shuffle=True, class_weight=None):

    N = X.shape[0]
    indices = np.arange(N)

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, N, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            X_batch = X[batch_idx].astype(np.float32)
            y_batch = y[batch_idx].astype(np.int32)

            if class_weight is not None:
                w0 = float(class_weight.get(0, 1.0))
                w1 = float(class_weight.get(1, 1.0))
                sample_weight = np.where(y_batch == 1, w1, w0).astype(np.float32)
                yield X_batch, y_batch, sample_weight  # ← (x, y, sample_weight)
            else:
                yield X_batch, y_batch  # ← 기존처럼 (x, y)


def pr_auc(y_true, y_prob):
    # sklearn average_precision_score == PR-AUC
    return average_precision_score(y_true, y_prob)


def best_f1_threshold(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    i = int(np.nanargmax(f1s))
    thr_best = float(thr[i]) if i < len(thr) else 0.5
    return float(f1s[i]), thr_best


def gru_build_and_compile(hp, seq_len, feature_len,
                      binary_idx, category_idx, feature_category_sizes):
    #print(tf.config.list_physical_devices('GPU'))
    """하이퍼파라미터 dict(hp)로 모델 생성·컴파일"""
    # ---- 모델 ----
    model, _ = gru_att_cls(
        seq_len=int(seq_len),
        feature_len=int(feature_len),
        binary_idx=binary_idx,
        category_idx=category_idx,
        category_emb_size=feature_category_sizes,
        emb_dim=hp["emb_dim"],
        gru_seq_len=hp["gru_seq_len"],
        gru_units=hp["gru_units"],
        dropout_rate=hp["dropout_rate"],
        att_units=hp["att_units"],
    )
    # ---- 컴파일 ----
    model.compile(
        optimizer=optimizers.Adam(learning_rate=hp["lr"]),
        loss=losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            metrics.AUC(name="PR-AUC", curve="PR"),
            metrics.AUC(name="ROC-AUC", curve="ROC"),
            metrics.Precision(name="Precision"),
            metrics.Recall(name="Recall"),
        ],
    )
    #model.summary()
    return model

def transformer_build_and_compile(hp, seq_len, feature_len,
                      binary_idx, category_idx, feature_category_sizes):
    model, _ = transformer_att_cls(
        seq_len=int(seq_len),
        feature_len=int(feature_len),
        binary_idx=binary_idx,
        category_idx=category_idx,
        category_emb_size=feature_category_sizes,
        emb_dim=hp["emb_dim"],
        num_heads = hp["num_heads"],
        key_dim = hp["key_dim"],
        dense_dim=int(hp["num_heads"] * hp["key_dim"]),
        encoder_n = hp["encoder_n"],
        dropout_rate=hp["dropout_rate"],
        att_units=hp["att_units"],
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=hp["lr"]),
        loss=losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            metrics.AUC(name="PR-AUC", curve="PR"),
            metrics.AUC(name="ROC-AUC", curve="ROC"),
            metrics.Precision(name="Precision"),
            metrics.Recall(name="Recall"),
        ],
    )
    return model


import json
from pathlib import Path
from typing import Dict, Any

def load_best_params_from_log(log_path: Path) -> Dict[str, Any]:
    best_rec = None
    best_value = -float("inf")

    with open(log_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            value = rec.get("value")
            state = rec.get("state")
            # 실패하거나 pruned된 trial은 걸러줄 수도 있음
            if value is None:
                continue
            if not str(state).endswith("COMPLETE"):
                continue
            if value > best_value:
                best_value = value
                best_rec = rec

    if best_rec is None:
        raise RuntimeError("로그에서 유효한 trial을 찾지 못했습니다.")

    return best_rec["params"]
