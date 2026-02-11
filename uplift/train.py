from pathlib import Path
from src.xgb_cls import train_xgb_cls
from src.xgb_reg import train_xgb_reg

ROOT_DIR = Path(__file__).resolve().parent.parent
config = {
    "data_load_dir": ROOT_DIR / "uplift" / "data",
    "n_folds": 3,
    "n_trials": 50,
    "n_jobs": 7,
    "sample_frac": 0.6,
}


# <editor-fold desc="CLS model train (outcome models)">
train_xgb_cls(model_save_dir = ROOT_DIR / "uplift" / "models" / "cls" / "conversion" /"treatment",
              treatment = True, outcome = "conversion", **config)

train_xgb_cls(model_save_dir = ROOT_DIR / "uplift" / "models" / "cls" / "conversion" /"control",
              treatment = False, outcome = "conversion", **config)
# </editor-fold>

# <editor-fold desc="REG model train (effect models)">
train_xgb_reg(model_save_dir = ROOT_DIR / "uplift" / "models" / "reg" / "conversion" /"treatment",
              model_load_dir = ROOT_DIR / "uplift" / "models" / "cls" / "conversion",
              treatment = True, outcome = "conversion", **config)

train_xgb_reg(model_save_dir = ROOT_DIR / "uplift" / "models" / "reg" / "conversion" /"control",
              model_load_dir = ROOT_DIR / "uplift" / "models" / "cls" / "conversion",
              treatment = False, outcome = "conversion", **config)
# </editor-fold>


