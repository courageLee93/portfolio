from src.data import *
ROOT_DIR = Path(__file__).resolve().parent.parent

# <editor-fold desc="raw data 분할 저장">
'''
# 나눠져 있는 사용자 데이터를 같은 버킷내에 들어가도록 분할
sharding_user(1_000_000, 10,
              ROOT_DIR / "dataset" / "criteo_attribution_dataset.tsv.gz",
              ROOT_DIR / "attribution" / "data" / "uid_buckets")
# 그룹화해서 인덱싱 & 인덱스 테이블 생성
indexing_groups(group_columns=["uid", "conversion_id", "campaign"],
                data_path=ROOT_DIR / "attribution" / "data" / "uid_buckets",
                save_path=ROOT_DIR / "attribution" / "data" / "grouped_uid_buckets")
'''
# </editor-fold>

# <editor-fold desc="test/train(N-fold) 분할">
'''
split_test_train_set(data_path=ROOT_DIR / "attribution" / "data" / "grouped_uid_buckets",
                     save_path=ROOT_DIR / "attribution" / "data" / "training_set",
                     fold_size=3, test_size=0.2, random_state=42)
'''
# </editor-fold>

# <editor-fold desc="train+val 데이터 생성">
'''
train_val_concatenate(data_path=ROOT_DIR/"attribution" / "data",
                      save_path=ROOT_DIR / "attribution" / "data" / "train")
'''
# </editor-fold>



# <editor-fold desc="float32 --> float16 위해 데이터 매핑 ">
'''
map = category_mapping(ROOT_DIR / "attribution" / "data" / "training_set" / "train",
                       ROOT_DIR / "attribution" / "data" / "training_set" / "test",
                       ROOT_DIR / "attribution" / "data" / "training_set",
                       ["campaign","cat1","cat2","cat3","cat4","cat5","cat6","cat7","cat8","cat9"])
'''
# </editor-fold>


# <editor-fold desc="속도 개선을 위해 파일형식 변경 & 변환 데이터로 저장 ">
parquet_to_memmap_with_padding(data_path=ROOT_DIR / "attribution" / "data" / "training_set" / "train",
                               mapping_path=ROOT_DIR / "attribution" / "data" / "training_set")
parquet_to_memmap_with_padding(data_path=ROOT_DIR / "attribution" / "data" / "training_set" / "test",
                               mapping_path=ROOT_DIR / "attribution" / "data" / "training_set")
# </editor-fold>


