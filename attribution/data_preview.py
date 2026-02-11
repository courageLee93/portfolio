import pandas as pd
from pathlib import Path
from collections import Counter


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "dataset"
DATA = DATA_DIR / "criteo_attribution_dataset.tsv.gz"
CHUNK_SIZE = 1_000_000

total_rows = 0
counts_user = Counter()
counts_campaign = Counter()
counts_cat = [Counter() for _ in range(9)]

counts_uid_conv_campaign = Counter()

reader = pd.read_csv(DATA,sep="\t", compression="gzip",chunksize=CHUNK_SIZE)
for i, chunk in enumerate(reader):

    n = len(chunk)
    total_rows += n

    user_count = chunk["uid"].value_counts()
    counts_user.update(user_count.to_dict())

    campaign_count = chunk["campaign"].value_counts()
    counts_campaign.update(campaign_count.to_dict())

    for j in range(9):
        cat_col = f"cat{j+1}"
        counts_cat[j].update(chunk[cat_col].value_counts().to_dict())

    tri = chunk.value_counts(["uid", "conversion_id", "campaign"]).to_dict()
    counts_uid_conv_campaign.update(tri)

    print(f"청크 {i} 완료")


df_count_user = pd.DataFrame.from_dict(counts_user, orient="index", columns=["count"])
df_count_campaign = pd.DataFrame.from_dict(counts_campaign, orient="index", columns=["count"])
df_count_cat = []
for i, counter in enumerate(counts_cat, start=1):
    df_cat = pd.DataFrame.from_dict(counter, orient="index", columns=["count"])
    df_cat.index.name = f"cat{i}"       # 인덱스 이름 지정
    df_cat.reset_index(inplace=True)    # index를 컬럼으로 변환
    df_count_cat.append(df_cat)
df_uid_conv_campaign = pd.DataFrame(
    [(uid, conv, camp, cnt) for (uid, conv, camp), cnt in counts_uid_conv_campaign.items()],
    columns=["uid", "conversion_id", "campaign", "count"]
)
df_conv_not = df_uid_conv_campaign[df_uid_conv_campaign["conversion_id"] == -1]
df_conv  = df_uid_conv_campaign[df_uid_conv_campaign["conversion_id"] != -1]


pd.set_option("display.float_format", "{:.2f}".format)
print(df_conv["count"].describe())
print((df_conv["count"]>100).sum())


import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.hist(df_conv["count"], bins=30, alpha=0.7, edgecolor="black")
plt.title("Distribution of 'count'")
plt.xlabel("count")
plt.ylabel("frequency")
# 평균선 추가
mean_val = df_conv["count"].mean()
plt.axvline(mean_val, color="red", linestyle="dashed", linewidth=2, label=f"Mean = {mean_val:.2f}")
plt.legend()
plt.show()
