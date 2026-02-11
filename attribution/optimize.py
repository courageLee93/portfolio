from src.model_train import *
from sklearn.model_selection import train_test_split, StratifiedKFold


ROOT_DIR = Path(__file__).resolve().parent.parent

print("data load ...")
data_path = ROOT_DIR / "data" / "train"
X = np.load(data_path / "x.npy")
y = np.load(data_path / "y.npy")

print("compute cw ...")
n_neg = np.sum(y == 0)
n_pos = np.sum(y == 1)
class_weight = {0: 1.0, 1: round(n_neg / n_pos)}

print("data split ...")
X_train, X_val, y_train, y_val = train_test_split(X, y,test_size=0.25, stratify=y,random_state=42)
del X, y
gc.collect()

print("optimization ...")
opt_path = ROOT_DIR / "opt"
opt_path.mkdir(parents=True, exist_ok=True)
study = optimize_model_numpy(
    X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
    model_name="gru",
    save_path= opt_path,
    class_weight=class_weight,
    batch_size=1024, epoch_num=10, n_trials=12,
)
print("save params ...")
best_path = ROOT_DIR / "best_params"
best_path.mkdir(parents=True, exist_ok=True)
best = {
    "best_params": study.best_params,
    "best_value(PR-AUC)": float(study.best_value),
    "best_user_attrs": dict(study.best_trial.user_attrs),
}
with open(best_path / "best_params.json", "w") as f:
    json.dump(best, f, indent=2, ensure_ascii=False)






