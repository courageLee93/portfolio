from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent

# <editor-fold desc="Effect Model">
m_t = joblib.load(ROOT_DIR / "uplift" / "models" / "reg" / "conversion" / "treatment" / "model.pkl")
m_c = joblib.load(ROOT_DIR / "uplift" / "models" / "reg" / "conversion" / "control" / "model.pkl")
# </editor-fold>

# <editor-fold desc="Data">
data = pd.read_parquet(ROOT_DIR / "uplift" / "data" / "test.parquet")
covar = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"]
X = data[covar]
y = data["conversion"]
# </editor-fold>

# <editor-fold desc="Propensity Score">
train_data = pd.read_parquet(ROOT_DIR / "uplift" / "data" / "train.parquet")
pi = float((train_data["treatment"] == 1).mean())
print(f"Propensity Score: {pi}")
# </editor-fold>

# <editor-fold desc="X-Learner">
pred_t = m_t.predict(X)
pred_c = m_c.predict(X)
uplift_score = (1 - pi) * pred_t + pi * pred_c
# save
out = data.copy()
out["uplift_score"] = uplift_score
out_path = ROOT_DIR / "uplift" / "result" / "conversion"
out_path.mkdir(parents=True, exist_ok=True)
out.to_parquet(out_path / "uplift.parquet", index=False)
# </editor-fold>
