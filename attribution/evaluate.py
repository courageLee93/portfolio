from src.model_evaluate import *


ROOT_DIR = Path(__file__).resolve().parent.parent


# <editor-fold desc="data load">
print("data load ...")
data_path = ROOT_DIR / "attribution" / "data" / "training_set" / "test"
X = np.load(data_path / "x.npy")
y = np.load(data_path / "y.npy")
# </editor-fold>



# <editor-fold desc="config">
config = {
    "seq_len": 376,
    "feature_len": 11,
    "binary_idx": [0],
    "category_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "category_emb_size": [675, 9, 70, 1829, 21, 51, 30, 57196, 11, 30],
}
gru_keras_config = {
    **config, "emb_dim": 16, "gru_seq_len": 1, "gru_units": 192,
    "dropout_rate": 0.009461849336318658, "att_units": 128,
}
gru_optuna_config = {
    **config, "emb_dim": 16, "gru_seq_len": 1, "gru_units": 192,
    "dropout_rate": 0.025118810817618964, "att_units": 128,
}
transformer_keras_config = {
    **config, "emb_dim": 48, "num_heads": 4, "key_dim": 32, "dense_dim": int(4*32),"encoder_n": 1,
    "dropout_rate": 0.03811382089917938, "att_units": 192
}
transformer_optuna_1_config = {
    **config, "emb_dim": 32, "num_heads": 2, "key_dim": 48, "dense_dim": int(2*48),"encoder_n": 1,
    "dropout_rate": 0.22196592801554627, "att_units": 192
}
transformer_optuna_2_config = {
    **config, "emb_dim": 32, "num_heads": 4, "key_dim": 48, "dense_dim": int(4*48),"encoder_n": 1,
    "dropout_rate": 0.1484132092131661, "att_units": 192
}
CONFIG_MAP = {
    ("gru", "keras"): gru_keras_config,
    ("gru", "optuna"): gru_optuna_config,
    ("transformer", "keras"): transformer_keras_config,
    ("transformer", "optuna_1"): transformer_optuna_1_config,
    ("transformer", "optuna_2"): transformer_optuna_2_config,
}
# </editor-fold>

# <editor-fold desc="prediction">
model_name = ["gru","transformer"]
for model in model_name:
    if model == "gru":
        optimize_method = ["keras","optuna"]
    else:
        optimize_method = ["keras","optuna_1","optuna_2"]
    for method in optimize_method:
        print(f"{model} : {method}")
        model_config = CONFIG_MAP[(model, method)]

        result = model_prediction(X=X, y=y,
                                  model_name=model, model_config=model_config,
                                  weight_path=ROOT_DIR / "attribution" / "model" / model / "model" / method,
                                  save_path=ROOT_DIR / "attribution" / "result" / model / method)
        model_evaluate(y_true=result["y_true"], y_pred=result["y_pred"], pred=result["pred"],
                       save_path=ROOT_DIR / "attribution" / "result" / model / method)

        result_short = result[result["seq_len"] < 50].reset_index(drop=True)
        model_evaluate(y_true=result_short["y_true"], y_pred=result_short["y_pred"], pred=result_short["pred"],
                       save_path=ROOT_DIR / "attribution" / "result" / model / method / "short")

        result_long = result[result["seq_len"] >= 50].reset_index(drop=True)
        model_evaluate(y_true=result_long["y_true"], y_pred=result_long["y_pred"], pred=result_long["pred"],
                       save_path=ROOT_DIR / "attribution" / "result" / model / method / "long")

# </editor-fold>
