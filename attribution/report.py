from src.model_evaluate import *

ROOT_DIR = Path(__file__).resolve().parent.parent

model_name = ["gru", "transformer"]
for model in model_name:
    if model == "gru":
        optimize_method = "optuna"
    else:
        optimize_method = "optuna_1"
    result_path = ROOT_DIR / "attribution" / "result" / f"{model}/{optimize_method}"
    result = np.load(result_path / "result.npy")  # 1629856
    result_short = result[result[:, 4] < 10]  # 1599088
    result_long = result[result[:, 4] >= 10]  # 30768

    save_path = ROOT_DIR / "attribution" / "reports" / f"{model}"

    model_evaluate(y_true=result[:,0],y_pred=result[:,1],pred=result[:,2],
                   save_path=save_path)
    model_evaluate(y_true=result_short[:,0],y_pred=result_short[:,1],pred=result_short[:,2],
                   save_path=save_path/"short")
    model_evaluate(y_true=result_long[:,0],y_pred=result_long[:,1],pred=result_long[:,2],
                   save_path=save_path/"long")




