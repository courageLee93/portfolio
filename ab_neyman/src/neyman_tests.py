import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep, proportion_effectsize
from pingouin import compute_effsize


def plot_distribution_density(df:pd.DataFrame, metric_col:str, bin_num:int):

    plot_data = df.copy()
    metric_label = metric_col.upper()

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    # 1) A vs B combined (density)
    sns.histplot(
        data=plot_data,
        x=metric_col,
        hue="group",
        bins=bin_num, stat="density",
        common_norm=False,
        alpha=0.5, ax=axes[0]
    )
    axes[0].set_title(f"{metric_label} Distribution (A vs B)")
    axes[0].set_ylabel("Density")
    # 2) A only
    sns.histplot(
        data=plot_data[plot_data["group"] == "A"],
        x=metric_col,
        bins=bin_num, stat="density",
        color="steelblue",
        alpha=0.6, ax=axes[1]
    )
    axes[1].set_title(f"{metric_label} Distribution (Group A)")
    axes[1].set_ylabel("Density")
    # 3) B only
    sns.histplot(
        data=plot_data[plot_data["group"] == "B"],
        x=metric_col,
        bins=bin_num, stat="density",
        color="orange",
        alpha=0.6, ax=axes[2]
    )
    axes[2].set_title(f"{metric_label} Distribution (Group B)")
    axes[2].set_ylabel("Density")
    # show
    plt.tight_layout()
    plt.show()


def t_sample_size(effect_size:float,alpha:float,power:float):
    # effect_size: cohen's d
    power_analysis = TTestIndPower()
    n_per_group = power_analysis.solve_power(effect_size=effect_size,
                                             alpha=alpha,
                                             power=power)
    print(f"{n_per_group} per group")
    return n_per_group


def t_power_curve(alpha: float = 0.05, power: float = 0.8):
    # d 범위 설정
    effect_sizes = np.linspace(0.1, 0.8, 30)  # 0.1 ~ 0.8 사이 30개 점

    n_per_group = [t_sample_size(d, alpha, power) for d in effect_sizes]

    plt.figure(figsize=(6, 4))
    plt.plot(effect_sizes, n_per_group, marker="o")
    plt.xlabel("Effect Size (Cohen's d)")
    plt.ylabel("Required Sample Size per Group (N)")
    plt.title(f"Welch t-test\nalpha={alpha}, power={power}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def t_test_welch(data_a:Series, data_b:Series, alpha: float = 0.05):
    a = np.asarray(data_a)
    b = np.asarray(data_b)

    # 효과크기
    mean_a, mean_b = a.mean(), b.mean()
    diff = mean_a - mean_b
    cd = compute_effsize(a, b, eftype='cohen')
    print("mean_a:", mean_a)
    print("mean_b:", mean_b)
    print("diff:", diff)
    print("Cohen's d:", cd)

    # 검정 (등분산 비가정)
    cm = CompareMeans(DescrStatsW(a), DescrStatsW(b))
    t, p, dfree = cm.ttest_ind(usevar='unequal')
    print("t-stat:", t)
    print("p-value:", p)
    print("df:", dfree)
    # 신뢰구간
    ci_low, ci_high = cm.tconfint_diff(alpha=alpha, usevar='unequal')
    print("ci_low:", ci_low)
    print("ci_high:", ci_high)

    result = {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "diff": diff,
        "cohen_d": cd,
        "t_stat": t,
        "p_value": p,
        "d_free": dfree,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
    return result


def z_sample_size(p_a:float, p_b:float,alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0):
    effect_size = proportion_effectsize(p_a, p_b)  # Cohen's h
    analysis = NormalIndPower()
    n1 = analysis.solve_power(effect_size=effect_size,
                              alpha=alpha,
                              power=power,
                              ratio=ratio,
                              alternative='two-sided')
    n2 = n1 * ratio
    print(f"n_a:{n1} / n_b:{n2}")
    return n1, n2, effect_size


def z_power_curve(alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0):
    effect_sizes_h = np.linspace(0.05, 0.5, 30)
    analysis = NormalIndPower()
    n1_list = [
        analysis.solve_power(effect_size=h,
                             alpha=alpha,
                             power=power,
                             ratio=ratio,
                             alternative='two-sided')
        for h in effect_sizes_h
    ]
    plt.figure(figsize=(6, 4))
    plt.plot(effect_sizes_h, n1_list, marker="o")
    plt.xlabel("Effect Size (Cohen's h)")
    plt.ylabel("Required Sample Size per Group (N)")
    plt.title(f"two-proportion z-test\nalpha={alpha}, power={power}, ratio={ratio}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def z_sample_size_by_uplift(p_a: float,
                              uplifts=(0.05, 0.1, 0.2, 0.3, 0.5),
                              alpha: float = 0.05,
                              power: float = 0.8,
                              ratio: float = 1.0):
    results = []
    for u in uplifts:
        p_b = p_a * (1 + u)  # 상대 uplift
        n1, n2, h = z_sample_size(p_a, p_b, alpha=alpha, power=power, ratio=ratio)
        results.append((u, p_b, n1, n2, h))

    # 출력
    print(f"Baseline = {p_a:.6f}")
    print(f"alpha={alpha}, power={power}, ratio={ratio}")
    for u, p_b, n1, n2, h in results:
        print(f"Uplift {u * 100:5.1f}% → p={p_b:.6f}, "
              f"N_per_group≈{n1:.1f}, h={h:.4f}")

    # 커브 그리기 (uplift% vs N_per_group)
    uplift_pct = [u * 100 for u, *_ in results]
    n_per_group = [n1 for _, _, n1, _, _ in results]

    plt.figure(figsize=(6, 4))
    plt.plot(uplift_pct, n_per_group, marker="o")
    plt.xlabel("Target Uplift in ICR (%)")
    plt.ylabel("Required Sample Size per Group (N)")
    plt.title(f"Sample Size vs Uplift\nbaseline={p_a:.4%}, alpha={alpha}, power={power}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def z_test_proportion(count_a:int, count_b:int, nob_a:int, nob_b:int, alpha: float = 0.05):
    counts = np.array([count_a, count_b])
    nobs = np.array([nob_a, nob_b])

    # 효과크기
    p_a = count_a/nob_a
    p_b = count_b/nob_b
    diff = p_b - p_a
    risk_ratio = p_b / p_a
    odds_ratio = (p_b / (1 - p_b)) / (p_a / (1 - p_a))
    ch = proportion_effectsize(p_a, p_b)
    print("p_a: ", p_a)
    print("p_b: ", p_b)
    print("diff:", diff)
    print("RR:", risk_ratio)
    print("OR:", odds_ratio)
    print("Cohen's h:", ch)

    # z-test
    z_stat, p_value = proportions_ztest(count=counts, nobs=nobs, alternative='two-sided')
    print("z_stat:", z_stat)
    print("p_value:", p_value)

    # 두 비율 차이의 신뢰구간
    ci_low, ci_high = confint_proportions_2indep(
        count1=count_a, nobs1=nob_a,
        count2=count_b, nobs2=nob_b,
        method='newcombe',
        compare='diff',
        alpha=alpha
    )
    print("ci_low:", ci_low)
    print("ci_high:", ci_high)

    result = {
        "p_a": p_a,
        "p_b": p_b,
        "diff": diff,
        "risk_ratio": risk_ratio,
        "odds_ratio": odds_ratio,
        "cohen_h": ch,
        "z_stat": z_stat,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
    return result

