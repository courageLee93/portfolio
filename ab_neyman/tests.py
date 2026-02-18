from pathlib import Path
from src.neyman_tests import *


# <editor-fold desc="dataset load">
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "dataset"
df = pd.read_csv(DATA_DIR / "fake_ab_test_2000.csv")
# </editor-fold>

# <editor-fold desc="RPI">
df["rpi"] = (df["revenue"] / df["impressions"]).astype(float)
df["log_rpi"] = np.log1p(df["rpi"])  # log(1 + rpi)

#plot_distribution_density(df,"rpi",50)
plot_distribution_density(df,"log_rpi",50)

a_log_rpi = df[df["group"] == "A"]["log_rpi"]
b_log_rpi = df[df["group"] == "B"]["log_rpi"]

#t_size_sample = t_sample_size(effect_size=0.2,alpha=0.05,power=0.8)
#t_power_curve(alpha=0.05,power=0.8)

t_result = t_test_welch(data_a=a_log_rpi, data_b=b_log_rpi, alpha=0.05)
# </editor-fold>


# <editor-fold desc="ICR">
a_conv = df[df["group"] == "A"]["conversions"].sum()
b_conv = df[df["group"] == "B"]["conversions"].sum()
a_imp = df[df["group"] == "A"]["impressions"].sum()
b_imp = df[df["group"] == "B"]["impressions"].sum()

p_a = 0.00423               # baseline ICR
target_uplift = 0.20        # +20%
p_b = p_a * (1 + target_uplift)  # 0.00423 * 1.2 ≈ 0.005076
#z_size_sample = z_sample_size(p_a, p_b, alpha=0.05, power=0.8, ratio=1.0)
#z_power_curve(alpha=0.05, power=0.8, ratio=1.0)
#z_sample_size_by_uplift(p_a)

z_result = z_test_proportion(a_conv, b_conv, a_imp, b_imp, alpha=0.05)
# </editor-fold>

