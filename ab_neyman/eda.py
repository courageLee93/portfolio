import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import json

# <editor-fold desc="dataset load">
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "dataset"
df = pd.read_csv(DATA_DIR / "fake_ab_test_2000.csv")
interval = 5
df["time_bucket"] = (df["time_on_site"] // interval).astype(int)
df["time_range"] = "[" + (df["time_bucket"] * interval).astype(str) + \
                   "," + ((df["time_bucket"] + 1) * interval).astype(str) + ")"
# </editor-fold>

save_path = ROOT_DIR / "ab_neyman" / "report" / "eda"
save_path.mkdir(parents=True, exist_ok=True)


# <editor-fold desc="distribution">
cols = ["impressions", "clicks", "revenue", "time_on_site", "pages_viewed"]
fig, axes = plt.subplots(1, len(cols), figsize=(4*len(cols), 4), sharey=False)
fig.suptitle("User-Level Behavioral Metric Distributions")
for i, col in enumerate(cols):
    sns.violinplot(data=df, x="group", y=col, ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel("Group")
    axes[i].set_ylabel(col)
plt.tight_layout()
plt.savefig(save_path / "metric_distribution.png", dpi=300, bbox_inches="tight")
# </editor-fold>

for group in ["A", "B"]:

    # <editor-fold desc="central tendency & cpu & ucr">
    df_desc = df.loc[df["group"] == group, cols].describe().T
    df_desc["IQR"] = df_desc["75%"] - df_desc["25%"]
    cpu = df.loc[df["group"] == group, "clicks"].sum() / (df["group"] == group).sum()
    ucr = (df.loc[df["group"] == group, "conversions"] > 0).mean() * 100

    df_desc_dict = df_desc.to_dict()
    df_desc_dict["cpu"] = float(cpu)
    df_desc_dict["ucr"] = float(ucr)
    with open(save_path/f"stats_group_{group}.json", "w") as f:
        json.dump(df_desc_dict, f, indent=4)
    # </editor-fold>

    # <editor-fold desc="Behavior-Conversion Relationship">
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))
    fig.suptitle(f"Group {group} - Behavior to Conversion", fontsize=15, y=1.02)
    # ---- 1) Clicks ---- #
    click_conv_rate = df[df["group"] == group].groupby("clicks").agg(
        users=("user_id", "nunique"),
        conversions=("conversions", "sum")
    )
    click_conv_rate["conv_rate"] = click_conv_rate["conversions"] / click_conv_rate["users"]
    axes[0].plot(click_conv_rate.index, click_conv_rate["conv_rate"], marker='o')
    axes[0].set_xlabel("Clicks")
    axes[0].set_ylabel("Conversion Rate")
    axes[0].set_title("Conversions vs Clicks")
    axes[0].grid(True)
    # ---- 2) Pages View ---- #
    pv_conv_rate = df[df["group"] == group].groupby("pages_viewed").agg(
        users=("user_id", "nunique"),
        conversions=("conversions", "sum")
    )
    pv_conv_rate["conv_rate"] = pv_conv_rate["conversions"] / pv_conv_rate["users"]
    axes[1].plot(pv_conv_rate.index, pv_conv_rate["conv_rate"], marker='o')
    axes[1].set_xlabel("Page Views")
    axes[1].set_ylabel("Conversion Rate")
    axes[1].set_title("Conversions vs Page Views")
    axes[1].grid(True)
    # ---- 3) Time Buckets ---- #
    time_conv_rate = df[df["group"] == group].groupby("time_range").agg(
        users=("user_id", "nunique"),
        conversions=("conversions", "sum")
    )
    time_conv_rate["conv_rate"] = time_conv_rate["conversions"] / time_conv_rate["users"]
    axes[2].plot(time_conv_rate.index, time_conv_rate["conv_rate"], marker='o')
    axes[2].set_xlabel("Time on Site")
    axes[2].set_ylabel("Conversion Rate")
    axes[2].set_title("Conversions vs Time on Site")
    axes[2].grid(True)
    # ---- save ---- #
    plt.tight_layout()
    plt.savefig(save_path / f"Group{group}-Behavior_to_Conversion.png", dpi=300, bbox_inches="tight")
    # </editor-fold>

    # <editor-fold desc="Behavior-Revenue Relationship (Scatter + Lowess Fit)">
    fig, ax = plt.subplots(1, 3, figsize=(20, 4))
    fig.suptitle(f"Group {group} - Behavior to Revenue", fontsize=15, y=1.02)
    # ---- 1) Clicks ---- #
    sns.regplot(data=df[df["group"] == group], x="clicks", y="revenue", lowess=True, scatter_kws={'alpha': 0.3}, ax=ax[0])
    ax[0].set_title("Revenue vs Clicks")
    ax[0].set_xlabel("Clicks")
    ax[0].set_ylabel("Revenue")
    # ---- 2) Pages View ---- #
    sns.regplot(data=df[df["group"] == group], x="pages_viewed", y="revenue", lowess=True, scatter_kws={'alpha': 0.3}, ax=ax[1])
    ax[1].set_title("Revenue vs Pages Viewed")
    ax[1].set_xlabel("Pages Viewed")
    # ---- 3) Time on Site ---- #
    sns.regplot(data=df[df["group"] == group], x="time_on_site", y="revenue", lowess=True, scatter_kws={'alpha': 0.3}, ax=ax[2])
    ax[2].set_title("Revenue vs Time on Site")
    ax[2].set_xlabel("Time (seconds)")
    # ---- save ---- #
    plt.tight_layout()
    plt.savefig(save_path / f"Group{group}-Behavior_to_Revenue.png", dpi=300, bbox_inches="tight")
    plt.close()
    # </editor-fold>


