import gc
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from numpy.lib.format import open_memmap


def sharding_user(chunk_size: int, bucket_size: int ,data_path: Path, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)
    reader = pd.read_csv(data_path, sep="\t", compression="gzip", chunksize=chunk_size)
    for i, chunk in enumerate(reader):
        chunk["bucket"] = chunk["uid"] % bucket_size
        for b, g in chunk.groupby("bucket", sort=False):
            f = save_path / f"bucket_{b:03d}.csv"
            g.drop(columns=["bucket"]).to_csv(f, index=False, mode="a", header=not f.exists())
        print(f"청크 {i} 샤딩 완료")
    # csv --> parquet
    for csv_path in save_path.glob("bucket_*.csv"):
        print(csv_path)
        df = pd.read_csv(csv_path)
        df = df.sort_values(["uid", "timestamp"], ascending=[True, True])
        pq_path = csv_path.with_suffix(".parquet")
        df.to_parquet(pq_path, index=False)
        csv_path.unlink()  # CSV 삭제
        print(f"{csv_path.name} → {pq_path.name} 변환/정렬 완료 (rows={len(df):,})")


def indexing_groups(group_columns:list,data_path: Path, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)
    index_df = pd.DataFrame(columns=["group_id", "label", "seq_len", "file"])
    group_id = 0  # 그룹 고유 ID
    for fname in sorted(data_path.glob("*.parquet")):
        print(fname)
        index_record = []
        df = pd.read_parquet(fname)
        if "group_id" not in df.columns:
            df["group_id"] = -1
        df["group_id"] = df["group_id"].astype("int64")
        grouped = df.groupby(group_columns, sort=False)
        num_groups = grouped.ngroups
        count_groups = 0
        for (uid, conversion_id, campaign), group in grouped:
            print(f"{fname.name}:{count_groups}/{num_groups}")
            df.loc[group.index, "group_id"] = group_id
            seq_len = len(group)
            label = 0 if conversion_id == -1 else 1
            index_record.append({
                "group_id": group_id,
                "label": label,
                "seq_len": seq_len,
                "file": fname.name,
            })
            group_id += 1
            count_groups += 1
        if index_record:
            index_df = pd.concat([index_df, pd.DataFrame(index_record)], ignore_index=True)
        df.to_parquet(save_path / f"{fname.name}", index=False)
        print(f"Saved: {fname.name}")
    print(f"Saved: index_df")
    index_df["group_id"] = index_df["group_id"].astype("int64")
    index_df["label"] = index_df["label"].astype("int8")
    index_df["seq_len"] = index_df["seq_len"].astype("int32")
    index_df.to_parquet(save_path / "index.parquet", index=False)


def split_test_train_set(data_path: Path, save_path: Path,
                         fold_size:int =3,
                         test_size: float = 0.2, random_state: int = 42):

    index_df = pd.read_parquet(data_path / "index.parquet")
    # train / test 분리 (라벨 비율 유지)
    train_df, test_df = train_test_split(
        index_df,
        test_size=test_size,
        stratify=index_df["label"],
        random_state=random_state,
    )

    # ---------- test set ----------
    y = test_df[["group_id", "label"]]
    x_list = []
    grouped = test_df.groupby("file", sort=False)
    for file, group in grouped:
        print(file)
        target_ids = set(group["group_id"])
        raw_df = pd.read_parquet(data_path / file)
        filtered_df = raw_df[raw_df["group_id"].isin(target_ids)]
        x_list.append(filtered_df)
    x = pd.concat(x_list, ignore_index=True)
    save_dir = save_path / "test"
    save_dir.mkdir(parents=True, exist_ok=True)
    y.to_parquet(save_dir / "test_y.parquet", index=False, compression="zstd")
    x.to_parquet(save_dir / "test_x.parquet", index=False, compression="zstd")

    # ---------- train set ----------
    if "fold_val" not in train_df.columns:
        train_df["fold_val"] = -1
    skf = StratifiedKFold(n_splits=fold_size, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
        val_index = train_df.iloc[val_idx].index
        train_df.loc[val_index, "fold_val"] = int(fold)
    for i in range(fold_size):
        for j in ["train", "val"]:
            print(f"fold:{i} - {j}")
            if j == "train":
                df = train_df[train_df["fold_val"] != i].drop("fold_val", axis=1)
            else:
                df = train_df[train_df["fold_val"] == i].drop("fold_val", axis=1)
            # y
            y = df[["group_id", "label"]]
            # x
            x_list = []
            grouped = df.groupby("file", sort=False)
            for file, group in grouped:
                print(file)
                target_ids = set(group["group_id"])
                raw_df = pd.read_parquet(data_path / file)
                filtered_df = raw_df[raw_df["group_id"].isin(target_ids)]
                x_list.append(filtered_df)
            x = pd.concat(x_list, ignore_index=True)
            save_dir = save_path / f"fold{i}"
            save_dir.mkdir(parents=True, exist_ok=True)
            y.to_parquet(save_dir / f"{j}_y.parquet", index=False, compression="zstd")
            x.to_parquet(save_dir / f"{j}_x.parquet", index=False, compression="zstd")


def train_val_concatenate(data_path: Path, save_path: Path):

    save_path.mkdir(parents=True, exist_ok=True)

    train_x_path = data_path / "training_set" / "fold0" / "train_x.parquet"
    train_y_path = data_path / "training_set" / "fold0" / "train_y.parquet"
    val_x_path = data_path / "training_set" / "fold0" / "val_x.parquet"
    val_y_path = data_path / "training_set" / "fold0" / "val_y.parquet"

    train_x = pd.read_parquet(train_x_path)
    val_x = pd.read_parquet(val_x_path)
    X_all = pd.concat([train_x, val_x], ignore_index=True)
    X_all.to_parquet(save_path / "train_x.parquet", index=False)

    train_y = pd.read_parquet(train_y_path)
    val_y = pd.read_parquet(val_y_path)
    y_all = pd.concat([train_y, val_y], ignore_index=True)
    y_all.to_parquet(save_path / "train_y.parquet", index=False)



def category_mapping(train_path, test_path, save_path, mapping_cols):
    train_x = pd.read_parquet(train_path / f"{train_path.name}_x.parquet", engine="pyarrow",
                              columns=["campaign", "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8",
                                       "cat9"])
    test_x = pd.read_parquet(test_path / f"{test_path.name}_x.parquet", engine="pyarrow",
                             columns=["campaign", "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8",
                                      "cat9"])
    all_x = pd.concat([train_x, test_x], ignore_index=True)
    mapping = {}  # 컬럼별 매핑 dict 저장
    for c in mapping_cols:
        # NaN 제거하고 유니크 값 추출
        uniques = all_x[c].dropna().unique()
        # 정렬해두면 재현성 ↑
        uniques_sorted = np.sort(uniques)
        # {원래값: 새 인덱스} 딕셔너리 생성 (1부터 시작)
        mapping[c] = {v: i + 1 for i, v in enumerate(uniques_sorted)}
        print(f"{c}: {len(uniques_sorted)} unique values → index 1 ~ {len(uniques_sorted)}")

    mapping_path = save_path / "col_mapping.pkl"
    joblib.dump(mapping, mapping_path)
    print("✅ mapping saved:", mapping_path)
    return mapping


def apply_category_mapping(df, mapping_path, mapping_cols, unk_index: int = 0):
    mapping = joblib.load(mapping_path / "col_mapping.pkl")
    for c in mapping_cols:
        m = mapping[c]
        df[c] = df[c].map(m).fillna(unk_index).astype("int32")
    return df


def parquet_to_memmap_with_padding(data_path:Path, mapping_path:Path):
    # ========== Data load ==========
    y = pd.read_parquet(data_path / f"{data_path.name}_y.parquet", engine="pyarrow",
                        columns=["group_id", "label"])
    X = pd.read_parquet(data_path / f"{data_path.name}_x.parquet", engine="pyarrow",
                        columns=["group_id", "timestamp", "click", "campaign",
                                 "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "cat9"])
    X = apply_category_mapping(X, mapping_path,
                               ["campaign","cat1","cat2","cat3","cat4","cat5","cat6","cat7","cat8","cat9"])
    y = y.sort_values("group_id", ascending=True)
    X = X.sort_values(["group_id", "timestamp"], ascending=[True, True])
    X_grouped = X.groupby("group_id", sort=False)
    del X; gc.collect()
    # ========== Data shape ==========
    data_len = y["group_id"].nunique()
    seq_len = 376
    feature_len = 11
    # ========== Padding define ==========
    padding_cols = {
        "click": -1.0, "campaign": 0,
        "cat1": 0, "cat2": 0, "cat3": 0, "cat4": 0, "cat5": 0,
        "cat6": 0, "cat7": 0, "cat8": 0, "cat9": 0
    }
    padding_row = np.array(list(padding_cols.values()), dtype="float16")
    padding = np.tile(padding_row, (seq_len, 1))
    # ========== mm file create ==========
    X_mm = open_memmap(data_path / "x.npy", dtype="float16", mode="w+", shape=(data_len, seq_len, feature_len))
    y_mm = open_memmap(data_path / "y.npy", dtype="int8", mode="w+", shape=(data_len,))
    # ========== parquet -> mm (+padding) ==========
    for index, row in enumerate(y.itertuples(index=False), start=0):
        group_id = row[0]
        label = row[1]
        print(f"{index}/{data_len}")
        #print(f"index: {index} - group_id: {group_id} - label: {label}")
        # ========== Y ==========
        y_mm[index] = np.int8(label)
        # ========== X ==========
        if group_id in X_grouped.indices:
            df = X_grouped.get_group(group_id)
            df = df.drop(columns=["group_id", "timestamp"]).to_numpy(dtype="float16")
            x_df = padding.copy()
            x_df[:df.shape[0]] = df
            X_mm[index] = x_df
        else:
            X_mm[index] = padding
    # ========== mm file save ==========
    X_mm.flush()
    y_mm.flush()

    # X = np.load(file_path / "x.npy", mmap_mode="r")
    # y = np.load(file_path / "y.npy", mmap_mode="r")