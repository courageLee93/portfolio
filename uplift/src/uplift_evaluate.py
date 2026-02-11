from pathlib import Path
import numpy as np
import pandas as pd
from sklift.metrics import qini_auc_score, uplift_auc_score, uplift_at_k, qini_curve, uplift_curve
from sklift.viz import plot_uplift_curve, plot_qini_curve
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def plt_qini_curve(uplift:pd.Series, outcome:pd.Series, treatment:pd.Series, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    plot_qini_curve(outcome, uplift, treatment)
    plt.title("Qini Curve")

    ax = plt.gca()
    # 과학적 표기(e-표기) 끄기
    ax.ticklabel_format(style='plain', axis='both', useOffset=False)
    # offset(+1e6 같은 보조표기) 완전히 제거
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    # (선택) 천 단위 콤마
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

    plt.savefig(save_path / "qini_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

def uplift_summary_all(uplift:pd.Series, outcome: pd.Series, treatment: pd.Series) -> dict:
    y = outcome.values
    t = treatment.values
    n_all = len(y)
    n_t = np.sum(t == 1)
    n_c = np.sum(t == 0)
    cr_t = y[t == 1].mean() if n_t > 0 else 0.0
    cr_c = y[t == 0].mean() if n_c > 0 else 0.0
    uplift_rate = cr_t - cr_c
    uplift_count = uplift_rate * n_t
    uplift_score_avg = uplift.mean()
    auuc = uplift_auc_score(outcome, uplift, treatment)
    qini = qini_auc_score(outcome, uplift, treatment)
    summary = {
        "k_frac": 1.0,                  # 전체 데이터 중 100%를 대상으로 평가했음을 의미 (Top-K가 아니라 전체 기준 요약)
        "k_n": int(n_all),              # 전체 데이터에 포함된 사용자 수 (실험에 참여한 전체 샘플 수)
        "n_t": int(n_t),                # 전체 사용자 중 treatment(처치군)에 속한 사용자 수
        "n_c": int(n_c),                # 전체 사용자 중 control(대조군)에 속한 사용자 수
        "cr_t": float(cr_t),            # 전체 treatment 그룹의 실제 전환율 = treatment 그룹에서 전환한 사용자 수 / treatment 전체 사용자 수
        "cr_c": float(cr_c),            # 전체 control 그룹의 실제 전환율 = control 그룹에서 전환한 사용자 수 / control 전체 사용자 수
        "uplift_rate": float(uplift_rate),   # 실험 전체 기준의 평균 처치 효과 = cr_t - cr_c
                                            # → "처치를 했을 때 전환율이 평균적으로 얼마나 더 높아졌는가"
                                            # → 실험의 실제 평균 효과(ATE 개념)
        "uplift_count": float(uplift_count),   # 실험 전체 기준의 증분 전환 수
                                            # = uplift_rate × treatment 사용자 수
                                            # → "처치로 인해 추가로 발생한 전환의 절대 개수"
        "uplift_score_avg": float(uplift_score_avg), # 모델이 예측한 개인별 uplift 점수의 평균값
                                   # → 실제 효과가 아니라, "모델이 보기에 평균적으로 얼마나 효과가 있을 것 같다고 예측했는가"
                                   # → 실험 결과와 직접 일치할 필요는 없음 (모델 출력의 분포 요약치)
        "auuc": float(auuc),     # Uplift AUC (Area Under Uplift Curve)
                                   # → 모델이 uplift가 큰 사람을 앞쪽에 잘 정렬했는지를 평가하는 지표
                                   # → 값이 클수록 "타겟팅 순서"가 좋다는 의미
                                   # → 실험 효과 크기와는 다른 개념 (정렬 성능 지표)
        "qini": float(qini),    # Qini AUC
                                   # → uplift 정렬 성능을 보는 또 다른 지표
                                   # → AUUC와 유사하지만, 기준선(baseline) 보정 방식이 다름
                                   # → 역시 "효과 크기"가 아니라 "정렬 품질"을 평가하는 지표
    }
    return summary

def uplift_summary_at_k(uplift:pd.Series, outcome:pd.Series, treatment:pd.Series, k_list:list, save_path: Path):
    N = len(outcome)
    order = np.argsort(-uplift)
    y = outcome[order]
    t = treatment[order]
    results = []
    for kf in k_list:
        k = int(N * kf)
        y_k = y[:k]
        t_k = t[:k]
        n_t = np.sum(t_k == 1)
        n_c = np.sum(t_k == 0)
        cr_t = y_k[t_k == 1].mean() if n_t > 0 else 0
        cr_c = y_k[t_k == 0].mean() if n_c > 0 else 0
        uplift_score_avg_k = float(uplift.iloc[order][:k].mean())
        results.append({
            "k_frac": kf, # 전체 데이터 중 상위 몇 %를 타겟했는지 (예: 0.1 = 상위 10%를 타겟)
            "k_n": k, # k_frac에 해당하는 실제 타겟된 사용자 수 (누적 타겟 수)
            "n_t": int(n_t),  # 상위 k_n명 안에 포함된 treatment(처치군) 사용자 수
            "n_c": int(n_c),  # 상위 k_n명 안에 포함된 control(대조군) 사용자 수
            "cr_t": float(cr_t),  # 상위 k_n명 중 treatment 그룹의 전환율 = (treatment 전환 수) / (treatment 전체 수)
            "cr_c": float(cr_c),  # 상위 k_n명 중 control 그룹의 전환율 = (control 전환 수) / (control 전체 수)
            "uplift_rate": float(cr_t - cr_c),  # 해당 구간에서의 순수 uplift 비율 = treatment 전환율 - control 전환율 → "처치가 전환율을 얼마나 더 올렸는가"
            "uplift_count": float((cr_t - cr_c) * n_t), # 해당 구간에서 발생한 누적 증분 전환 수 = uplift_rate × 해당 구간의 treatment 수 → "이만큼 더 전환을 만들어냈다"라는 절대 수치
            "uplift_score_avg_k": float(uplift_score_avg_k)
        })
    save_path.mkdir(parents=True, exist_ok=True)
    uplift_k = [uplift_at_k(outcome, uplift, treatment, strategy='by_group', k=k)
                for k in k_list]
    plt.figure(figsize=(6, 4))
    plt.plot([k * 100 for k in k_list], uplift_k, marker='o')
    plt.xlabel("Top K (%)")
    plt.ylabel("Uplift")
    plt.title("Uplift@K Curve")
    plt.grid(True)
    plt.savefig(save_path / "uplift_k_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    return pd.DataFrame(results)

def plt_uplift_curve(uplift:pd.Series, outcome:pd.Series, treatment:pd.Series, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    plot_uplift_curve(outcome, uplift, treatment)
    plt.title("Uplift Curve (AUUC)")

    ax = plt.gca()
    # 과학적 표기(e-표기) 끄기
    ax.ticklabel_format(style='plain', axis='both', useOffset=False)
    # offset(+1e6 같은 보조표기) 완전히 제거
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    # (선택) 천 단위 콤마
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

    plt.savefig(save_path / "uplift_curve(auuc).png", dpi=300, bbox_inches="tight")
    plt.close()

def plt_uplift_curve_k(uplift:pd.Series, outcome:pd.Series, treatment:pd.Series, k_frac:float, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)
    N = len(outcome)
    k = int(N * k_frac)
    # ---------- Model curve ----------
    order = np.argsort(-uplift)
    y_m = outcome[order]
    t_m = treatment[order]
    cum_t_m = np.cumsum(t_m)
    cum_c_m = np.cumsum(1 - t_m)
    cum_y_t_m = np.cumsum(y_m * t_m)
    cum_y_c_m = np.cumsum(y_m * (1 - t_m))
    rate_t_m = cum_y_t_m / np.maximum(cum_t_m, 1)
    rate_c_m = cum_y_c_m / np.maximum(cum_c_m, 1)
    inc_m = (rate_t_m - rate_c_m) * cum_t_m
    # ---------- Random baseline ----------
    rng = np.random.default_rng(42)
    order_r = rng.permutation(N)
    y_r = outcome[order_r]
    t_r = treatment[order_r]
    cum_t_r = np.cumsum(t_r)
    cum_c_r = np.cumsum(1 - t_r)
    cum_y_t_r = np.cumsum(y_r * t_r)
    cum_y_c_r = np.cumsum(y_r * (1 - t_r))
    rate_t_r = cum_y_t_r / np.maximum(cum_t_r, 1)
    rate_c_r = cum_y_c_r / np.maximum(cum_c_r, 1)
    inc_r = (rate_t_r - rate_c_r) * cum_t_r
    # ---------- Top-k slice ----------
    x = np.arange(1, N + 1)
    x_top = x[:k]
    inc_m_top = inc_m[:k]
    inc_r_top = inc_r[:k]
    # y-axis auto scale
    y_max = max(np.max(inc_m_top), np.max(inc_r_top)) * 1.05
    # ---------- Plot ----------
    plt.figure(figsize=(12, 8))
    plt.plot(x_top, inc_m_top, label="Model", linewidth=2)
    plt.plot(x_top, inc_r_top, label="Random", linestyle="--")
    plt.title(f"Top {k_frac*100:.1f}% Uplift Curve (AUUC)")
    plt.xlabel("Number targeted")
    plt.ylabel("Number of incremental outcome")
    plt.xlim(0, k)
    plt.ylim(0, y_max)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_path / f"uplift_{k_frac*100:.1f}%_curve(auuc).png", dpi=300, bbox_inches="tight")
    plt.close()

def plt_treat_control_rates_by_decile(uplift:pd.Series, outcome:pd.Series, treatment:pd.Series, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)
    # Treatment–Control Rate by Decile
    df = pd.DataFrame({
        "y": outcome.astype(int),
        "t": treatment.astype(int),
        "score": uplift
    }).dropna()
    # 상위 점수부터 10등분(동률 안전하게 처리하려면 rank 사용) (업리프트 점수 기준으로 10등분)
    df["decile"] = pd.qcut(df["score"].rank(method="first", ascending=False),
                           q=10, labels=[f"D{i}" for i in range(1, 11)])
    grp = df.groupby("decile")

    # 각 Decile에서 Treatment / Control 전환율 계산
    # 이 구간에서 Treatment 받은 사람들의 실제 전환율
    rate_t = grp.apply(lambda g: g.loc[g["t"] == 1, "y"].mean() if (g["t"] == 1).any() else 0.0)
    # 이 구간에서 Control 사람들의 실제 전환율
    rate_c = grp.apply(lambda g: g.loc[g["t"] == 0, "y"].mean() if (g["t"] == 0).any() else 0.0)
    # Treatment – Control = 실제 uplift
    uplift_by_decile = rate_t - rate_c

    # 막대그래프 (상위 decile이 왼쪽에 오도록 순서 뒤집기)
    order = [f"D{i}" for i in range(1, 11)]  # D1=상위 10%
    x = np.arange(len(order))
    plt.figure(figsize=(12, 8))
    plt.bar(x - 0.2, rate_t.reindex(order).values, width=0.4, label="P(y=1|T=1)")
    plt.bar(x + 0.2, rate_c.reindex(order).values, width=0.4, label="P(y=1|T=0)")
    plt.plot(x, uplift_by_decile.reindex(order).values, marker='o', linewidth=1.5, color="black", label="Uplift (T–C)")
    plt.xticks(x, order)
    plt.xlabel("Decile (D1=Top 10%)")
    plt.ylabel("Rate / Uplift")
    plt.title("Treatment–Control Rate by Decile")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    plt.savefig(save_path / "treatment_control_rate_decile.png", dpi=300, bbox_inches="tight")
    plt.close()

def plt_treat_control_rates_by_k(uplift:pd.Series, outcome:pd.Series, treatment:pd.Series, k_list:list, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "y": outcome.astype(int),
        "t": treatment.astype(int),
        "score": uplift
    }).dropna()

    # uplift 점수 내림차순 정렬
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    N = len(df)

    # 각 k에 대해 누적 Top-K 구간의 rate 계산
    rate_t_list, rate_c_list, uplift_list = [], [], []
    k_pct_labels = []

    for kf in k_list:
        k = int(N * float(kf))
        k = max(k, 1)  # 최소 1개는 포함

        top = df.iloc[:k]
        t_mask = (top["t"] == 1)
        c_mask = (top["t"] == 0)

        n_t = int(t_mask.sum())
        n_c = int(c_mask.sum())

        rate_t = float(top.loc[t_mask, "y"].mean()) if n_t > 0 else 0.0
        rate_c = float(top.loc[c_mask, "y"].mean()) if n_c > 0 else 0.0
        up = rate_t - rate_c

        rate_t_list.append(rate_t)
        rate_c_list.append(rate_c)
        uplift_list.append(up)
        k_pct_labels.append(f"{int(round(kf * 100))}%")

    x = np.arange(len(k_list))
    width = 0.2
    offset = width / 2

    plt.figure(figsize=(12, 8))
    plt.bar(x - offset, rate_t_list, width=width, label="P(y=1|T=1)")
    plt.bar(x + offset, rate_c_list, width=width, label="P(y=1|T=0)")
    plt.plot(x, uplift_list, marker="o", linewidth=1.5, color="black", label="Uplift (T–C)")

    plt.xticks(x, k_pct_labels)

    plt.xlabel("Top K (%) (cumulative)")
    plt.ylabel("Rate / Uplift")
    plt.title("Treatment–Control Rate by Top-K (Cumulative)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    plt.savefig(save_path / "treatment_control_rate_k.png", dpi=300, bbox_inches="tight")
    plt.close()