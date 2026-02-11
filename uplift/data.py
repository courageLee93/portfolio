from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "dataset"
SAVE_DIR = ROOT_DIR / "uplift" / "data"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(DATA_DIR / "criteo-research-uplift-v2.1.csv.gz", sep=",", compression="gzip")

# (visit, conversion) 3상태 stratify
# 0: no-visit (v=0,c=0)
# 1: visit-no-conv (v=1,c=0)
# 2: visit-conv (v=1,c=1)
T_COL = "treatment"
V_COL = "visit"
C_COL = "conversion"
split = np.select(
    [
        (data[V_COL] == 0) & (data[C_COL] == 0),
        (data[V_COL] == 1) & (data[C_COL] == 0),
        (data[V_COL] == 1) & (data[C_COL] == 1),
    ],
    [0,1,2]
)
# treatment까지 결합해 6개의 stratify
data["split"] = data[T_COL].astype(str) + "_" + split.astype(str)
print(data["split"].value_counts())

# train 70% / test 30%
train_idx, test_idx = train_test_split(
    data.index,
    test_size=0.3,
    random_state=42,
    stratify=data["split"]
)
train = data.loc[train_idx]
test  = data.loc[test_idx]
def show_strata_ratio(name, d):
    print(f"\n[{name}]")
    print(
        d["split"]
        .value_counts(normalize=True)
        .sort_index()
    )
# 전체 분포
show_strata_ratio("FULL", data)
# 분리 결과
show_strata_ratio("TRAIN", train)
show_strata_ratio("TEMP", test)

# save
train.to_parquet(SAVE_DIR / "train.parquet", index=False)
test.to_parquet(SAVE_DIR / "test.parquet", index=False)
np.save(SAVE_DIR / "train_idx.npy", train_idx.values)
np.save(SAVE_DIR / "test_idx.npy", test_idx.values)
