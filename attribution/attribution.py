from pathlib import Path
import numpy as np
import pandas as pd
from src.model_evaluate import model_attention_weight


ROOT_DIR = Path(__file__).resolve().parent.parent

#gru-optuna
result_path = ROOT_DIR / "attribution" / "result" / "gru" / "optuna" / "result.npy"
alpha_path = ROOT_DIR / "attribution" / "result" / "gru" / "optuna" / "alpha.npy"
gru_result = np.load(result_path)
gru_alpha = np.load(alpha_path)

#transformer-optuna_1
result_path = ROOT_DIR / "attribution" / "result" / "transformer" / "optuna_1" / "result.npy"
alpha_path = ROOT_DIR / "attribution" / "result" / "transformer" / "optuna_1" / "alpha.npy"
transformer_result = np.load(result_path)
transformer_alpha = np.load(alpha_path)


y_true = gru_result[:, 0].astype(int)
gru_collect = gru_result[:, 3].astype(int)
transformer_collect = transformer_result[:, 3].astype(int)
seq_len = gru_result[:, 4].astype(int)
result = pd.DataFrame({
    "y_true": y_true,
    "gru_collect": gru_collect,
    "transformer_collect": transformer_collect,
    "seq_len": seq_len,
})
result = result[result["y_true"] == 1]
result = result[(result["gru_collect"] == 1) & (result["transformer_collect"] == 1)]

result_short = result[result["seq_len"]==10]
result_long = result[result["seq_len"]>20]

'''
for idx, row in result_short.iterrows():
    print(idx)
    model_attention_weight(gru_alpha, idx, row["seq_len"],
        ROOT_DIR / "attribution" / "reports" / "gru" / "att" / "short")
    model_attention_weight(transformer_alpha, idx, row["seq_len"],
        ROOT_DIR / "attribution" / "reports" / "transformer" / "att" / "short")

for idx, row in result_long.iterrows():
    print(idx)
    model_attention_weight(gru_alpha, idx, row["seq_len"],
        ROOT_DIR / "attribution" / "reports" / "gru" / "att" / "long")
    model_attention_weight(transformer_alpha, idx, row["seq_len"],
        ROOT_DIR / "attribution" / "reports" / "transformer" / "att" / "long")
'''

result_too_long = result[result["seq_len"]>40]
for idx, row in result_too_long.iterrows():
    print(idx)
    model_attention_weight(gru_alpha, idx, row["seq_len"],
        ROOT_DIR / "attribution" / "reports" / "gru" / "att" / "long_2")
    model_attention_weight(transformer_alpha, idx, row["seq_len"],
        ROOT_DIR / "attribution" / "reports" / "transformer" / "att" / "long_2")