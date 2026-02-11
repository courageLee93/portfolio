from pathlib import Path
from .model_defs import *
from .model_utils import *
import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable



def model_prediction(
        X: np.ndarray, y: np.ndarray,
        model_name:str, model_config:dict,
        weight_path:Path, save_path:Path):
    # pred
    if model_name == "gru":
        _, attn_model = gru_att_cls(**model_config)
    else:
        _, attn_model = transformer_att_cls(**model_config)
    attn_model.load_weights(weight_path / "best_weights.weights.h5")
    batch_size = 1024
    batch_gen = numpy_batch_generator(X, y, batch_size, shuffle=False, class_weight=None)
    batch_steps = int(np.ceil(len(y) / batch_size))
    pred, alpha = attn_model.predict(batch_gen, steps=batch_steps, verbose=1)
    pred = pred.ravel()
    # result
    _, thr = best_f1_threshold(y, pred)
    y_pred = (pred >= thr).astype(int)
    y_collect = (y == y_pred).astype(int)
    result = pd.DataFrame({
        "y_true": y,
        "y_pred": y_pred,
        "pred": pred,
        "collect": y_collect,
        "seq_len": np.sum(X[..., 0] != -1, axis=1),
    })
    save_path.mkdir(parents=True, exist_ok=True)
    np.save(save_path / "result.npy", result)
    np.save(save_path / "alpha.npy", alpha)
    return result



def model_evaluate(
        y_true: pd.Series, y_pred: pd.Series, pred:pd.Series,
        save_path:Path):

    save_path.mkdir(exist_ok=True, parents=True)

    # <editor-fold desc="metrics">
    # pos_rate
    n_neg = np.sum(y_true == 0);
    n_pos = np.sum(y_true == 1)
    pos_rate = n_pos / (n_neg + n_pos) * 100
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, pred)
    # Log Loss
    lloss = log_loss(y_true, pred)
    # pr_auc
    #pr_auc = average_precision_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, pred)
    # f1
    f1 = f1_score(y_true, y_pred)
    # precision
    precision = precision_score(y_true, y_pred)
    # recall
    recall = recall_score(y_true, y_pred)
    # metrics
    metrics_dict = {
        "pos_rate": float(pos_rate),
        "roc_auc": float(roc_auc),
        "log_loss": float(lloss),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    # save
    with open(save_path / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
    # </editor-fold>

    # <editor-fold desc="confusion matrix">
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,  # 숫자 표시
        fmt="d",  # 정수 표시
        cmap="Blues",
        vmax=200000,  # 색상 상한 → 작은 값들도 보이게 튜닝
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.savefig(save_path / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    # </editor-fold>

    # <editor-fold desc="pr-curve & threshold vs f1 Score">
    pr, re, thrs = precision_recall_curve(y_true, pred)
    plt.figure(figsize=(6, 5))
    plt.plot(re, pr)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.savefig(save_path / "pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    f1_scores = 2 * (pr * re) / (pr + re)
    plt.figure(figsize=(6, 5))
    plt.plot(thrs, f1_scores[:-1])
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("Threshold vs F1")
    plt.grid(True)
    plt.savefig(save_path / "thr_f1.png", dpi=300, bbox_inches="tight")
    plt.close()
    # </editor-fold>

    # <editor-fold desc="roc-curve">
    fpr, tpr, thrs = roc_curve(y_true, pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={roc_auc:.4f})")
    plt.grid(True)
    plt.savefig(save_path / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    # </editor-fold>


def model_attention_weight(alpha:np.ndarray, index:int, seq_len:int, save_path:Path):

    save_path.mkdir(exist_ok=True, parents=True)

    alpha_i = np.squeeze(alpha[index])
    alpha_trim = alpha_i[:seq_len]
    x = np.arange(seq_len)  # 실제 time step
    '''
    # attention plot
    plt.figure(figsize=(14,4))
    plt.plot(alpha_trim)
    plt.title(f"Attention Weights (Sample {index})")
    plt.xlabel("Time Step")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.savefig(save_path / f"{index}_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # attention heatmap
    plt.figure(figsize=(14, 4))
    sns.heatmap(alpha_trim.reshape(1, -1), cmap="viridis")
    plt.title(f"Attention Heatmap (Sample {index})")
    plt.xlabel("Time Step")
    plt.savefig(save_path / f"{index}_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    '''


    # --- GridSpec으로 colorbar 공간을 "미리" 확보 (핵심) ---
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[3, 1],
        width_ratios=[1, 0.025],   # 오른쪽에 colorbar 전용 컬럼
        wspace=0.05, hspace=0.15
    )

    ax_line = fig.add_subplot(gs[0, 0])
    ax_hm   = fig.add_subplot(gs[1, 0], sharex=ax_line)
    cax     = fig.add_subplot(gs[1, 1])  # colorbar는 아래 행에만 (원하면 gs[:,1]로 두 행 span 가능)

    # 1) Line
    ax_line.plot(x, alpha_trim)
    ax_line.set_title(f"Attention Weights (Sample {index})")
    ax_line.set_ylabel("Weight")
    ax_line.grid(True)
    ax_line.set_xlim(0, seq_len - 1)
    plt.setp(ax_line.get_xticklabels(), visible=False)  # 위쪽 x 라벨 숨김(sharex)

    # 2) Heatmap (imshow로 좌표계 정확히 일치)
    im = ax_hm.imshow(
        alpha_trim.reshape(1, -1),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        extent=[-0.5, seq_len - 0.5, 0, 1]
    )
    ax_hm.set_yticks([])
    ax_hm.set_xlabel("Time Step")
    ax_hm.set_xlim(0, seq_len - 1)

    # 3) Time step tick "생략 방지" (필요하면 전부 표시)
    #    seq_len이 커지면 가독성 때문에 간격을 두는 게 보통이라, 아래처럼 자동 간격 추천
    step = 1 if seq_len <= 50 else (2 if seq_len <= 100 else 5)
    ticks = np.arange(0, seq_len, step)
    ax_hm.set_xticks(ticks)
    ax_hm.set_xticklabels([str(t) for t in ticks])

    # 4) Colorbar (이제 폭이 plot에 영향 없음)
    fig.colorbar(im, cax=cax)

    fig.savefig(save_path / f"{index}_attention.png", dpi=300, bbox_inches="tight")
    plt.close(fig)