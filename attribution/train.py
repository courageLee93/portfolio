from src.model_train import *

ROOT_DIR = Path(__file__).resolve().parent.parent

print("data load ...")
data_path = ROOT_DIR / "data" / "train"
X = np.load(data_path / "x.npy")
y = np.load(data_path / "y.npy")

print("compute cw ...")
n_neg = np.sum(y == 0)
n_pos = np.sum(y == 1)
class_weight = {0: 1.0, 1: round(n_neg / n_pos)}

print("hyper params load ...")
#log_path = opt_path / "trial_log.jsonl"
#best_hp = load_best_params_from_log(log_path)
#params_path  = ROOT_DIR / "best_params"
#with open(params_path / "best_params.json", "r") as f:
#    best = json.load(f)
#hp = best["best_params"]

gru_best_params_optuna = {
    "emb_dim": 16, "gru_units": 192, "gru_seq_len": 1,
    "dropout_rate": 0.025118810817618964, "att_units": 128, "lr": 0.001349822491576991
}
gru_best_params_keras = {
    "emb_dim": 16, "gru_units": 192, "gru_seq_len": 1,
    "dropout_rate": 0.009461849336318658, "att_units": 128, "lr": 0.0013905659147591836
}
trans_best_params_keras = {
    "emb_dim": 48, "num_heads": 4, "key_dim": 32, "encoder_n": 1,
    "dropout_rate": 0.03811382089917938, "att_units": 192, "lr": 0.00036263530626483027
}
trans_best_params_optuna_1 = {
    "emb_dim": 32, "num_heads": 2, "key_dim": 48, "encoder_n": 1,
    "dropout_rate": 0.22196592801554627, "att_units": 192, "lr": 0.0009521582074473866
}
trans_best_params_optuna_2 = {
    "emb_dim": 32, "num_heads": 4, "key_dim": 48, "encoder_n": 1,
    "dropout_rate": 0.1484132092131661, "att_units": 192, "lr": 0.0008271179635851826
}


print("gru train model ...")
model = train_model_numpy(
        X_train = X, y_train = y,
        model_name = "gru", hyper_params = hp,
        save_path = ROOT_DIR / "model" / "gru",
        class_weight = class_weight,
        batch_size = 1024, epoch_num = 50
)
tf.keras.backend.clear_session()

