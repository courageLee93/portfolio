from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (average_precision_score, log_loss, roc_auc_score,
                             mean_squared_error, mean_absolute_error)

ROOT_DIR = Path(__file__).resolve().parent.parent

# <editor-fold desc="model load">
m_cls_t = joblib.load(ROOT_DIR / "uplift" / "models" / "cls" / "conversion" / "treatment" / "model.pkl")
m_cls_c = joblib.load(ROOT_DIR / "uplift" / "models" / "cls" / "conversion" / "control" / "model.pkl")
m_reg_t = joblib.load(ROOT_DIR / "uplift" / "models" / "reg" / "conversion" / "treatment" / "model.pkl")
m_reg_c = joblib.load(ROOT_DIR / "uplift" / "models" / "reg" / "conversion" / "control" / "model.pkl")
# </editor-fold>

# <editor-fold desc="data load">
data = pd.read_parquet(ROOT_DIR / "uplift" / "data" / "test.parquet")
covar = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"]
# treatment
treatment = data[data["treatment"] == 1]
# control
control = data[data["treatment"] == 0]
# </editor-fold>

# <editor-fold desc="model & data">
datasets = {
    "treatment": {
        "X": treatment[covar],
        "y": treatment["conversion"],
        "p": treatment["conversion"] - m_cls_c.predict_proba(treatment[covar])[:, 1],
        "cls_model": m_cls_t,
        "reg_model": m_reg_t,
    },
    "control": {
        "X": control[covar],
        "y": control["conversion"],
        "p": m_cls_t.predict_proba(control[covar])[:, 1] - control["conversion"],
        "cls_model": m_cls_c,
        "reg_model": m_reg_c,
    }
}
# </editor-fold>

# <editor-fold desc="evaluate cls & reg model">
results = {}
for name, obj in datasets.items():

    print(name)
    X = obj["X"]
    y = obj["y"]
    p = obj["p"]
    cls_model = obj["cls_model"]
    reg_model = obj["reg_model"]

    # <editor-fold desc="cls evaluate">
    proba = cls_model.predict_proba(X)[:,1]
    # pos_rate
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    pos_rate = n_pos / (n_neg + n_pos) * 100
    # pr-auc
    pr_auc = float(average_precision_score(y, proba))
    # roc-auc
    roc_auc = roc_auc_score(y, proba)
    # log loss
    ll = float(log_loss(y, proba))
    # </editor-fold>

    # <editor-fold desc="reg evaluate">
    pred = reg_model.predict(X)
    rmse = np.sqrt(mean_squared_error(p, pred))
    mae = mean_absolute_error(p, pred)
    # </editor-fold>

    results[name] = {
        "cls":{
            "pos_rate": pos_rate,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "ll": ll,
        },
        "reg":{
            "rmse": rmse,
            "mae": mae,
        }
    }
# </editor-fold>
print(results)

# save report
"""
import json
def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    else:
        return obj
report = to_python(results)
report_path = ROOT_DIR / "uplift" / "report" / "conversion" / "models"
report_path.mkdir(parents=True, exist_ok=True)
with open(report_path / "report.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
"""