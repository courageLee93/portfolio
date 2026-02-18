from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
from src.beysian_tests import *

# <editor-fold desc="raw data">
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "dataset"
asos = pd.read_parquet(DATA_DIR / "asos_digital_experiments_dataset.parquet")
asos_summary = (
    asos.groupby(['experiment_id','variant_id','metric_id'])
        .agg(row_count=('experiment_id', 'size'))
        .reset_index()
)
# </editor-fold>

# <editor-fold desc="data choose">
exp_id = "a4386f"
var_id = 1
df = asos[
    (asos['experiment_id'] == exp_id)
    &
    (asos['variant_id'] == var_id)
].sort_values('time_since_start', ascending=True).reset_index(drop=True)
df_binary = df[df['metric_id'] == 1].reset_index(drop=True)
df_continuous = df[df['metric_id'] == 4].reset_index(drop=True)
# </editor-fold>

stop_rule = 2 #1,2
sample_num = 50000
rope_binary = 0.001
rope_continuous = 0.1


stop = False
winner = None
# <editor-fold desc="binary logic">
for t in range(len(df_binary)):
    row = df_binary.iloc[t]
    # A (k: success / f: fail) ※ cumulative
    k_a = row["mean_c"] * row["count_c"]
    f_a = row["count_c"] - k_a
    # B (k: success / f: fail) ※ cumulative
    k_b = row["mean_t"] * row["count_t"]
    f_b = row["count_t"] - k_b
    # posterior
    #post_a = beta_posterior(int(round(k_a)), int(round(f_a)))
    #post_b = beta_posterior(int(round(k_b)), int(round(f_b)))
    post_a = beta_posterior(k_a, f_a)
    post_b = beta_posterior(k_b, f_b)
    # sampling (monte carlo)
    sample_a = sample_beta_posterior(post_a, sample_num)
    sample_b = sample_beta_posterior(post_b, sample_num)
    # effect (b based)
    delta = sample_b - sample_a
    prob_positive = np.mean(delta > 0) # posterior probability
    prob_b_practical = np.mean(delta > rope_binary) # rope
    prob_a_practical = np.mean(delta < -rope_binary) # rope
    hdi_low, hdi_high = hdi_of_samples(delta) # credible interval (HDI)

    # stop rule
    if stop_rule == 1:
        if prob_positive >= 0.95 and hdi_low > 0 :
            stop = True
            winner = 'B'
        elif prob_positive <= 0.05 and hdi_high < 0 :
            stop = True
            winner = 'A'
    else:
        if prob_b_practical >= 0.95 and hdi_low > rope_binary :
            stop = True
            winner = 'B'
        elif prob_a_practical >= 0.95 and hdi_high < -rope_binary :
            stop = True
            winner = 'A'

    # results
    if (stop == True) or (t == len(df_binary)-1):
        # <editor-fold desc="result">
        print("<<Binary Results>>")
        print(f"[trial: {t} | winner: {winner}]")
        print(f"prob_positive=P(B>A): {prob_positive:.6f}")
        print(f"prob_b_practical=P(Δ>+{rope_binary}): {prob_b_practical:.6f}")
        print(f"prob_a_practical=P(Δ<-{rope_binary}): {prob_a_practical:.6f}")
        print(f"hdi_low, hdi_high: {hdi_low},{hdi_high}")
        # </editor-fold>
        # <editor-fold desc="plot(post_a vs post_b)">
        plt.figure(figsize=(8, 5))
        lo = min(np.percentile(sample_a, 0.5), np.percentile(sample_b, 0.5))
        hi = max(np.percentile(sample_a, 99.5), np.percentile(sample_b, 99.5))
        x = np.linspace(lo, hi, 500)
        plt.plot(x, gaussian_kde(sample_a)(x), label="A")
        plt.plot(x, gaussian_kde(sample_b)(x), label="B")
        plt.xlabel("Conversion rate")
        plt.ylabel("Density")
        plt.title("Posterior distributions of A and B")
        plt.legend()
        plt.tight_layout()
        plt.show()
        # </editor-fold>
        # <editor-fold desc="plot(delta)">
        plt.figure(figsize=(8, 5))
        lo = np.percentile(delta, 0.5)
        hi = np.percentile(delta, 99.5)
        x = np.linspace(lo, hi, 500)
        plt.plot(x, gaussian_kde(delta)(x), label="Delta")
        # 0 기준선
        plt.axvline(0, color="black", linestyle="--", linewidth=1, label="No effect (Δ = 0)")
        if stop_rule == 2:
            plt.axvline(rope_binary, color="gray", linestyle=":", linewidth=1, label=f"ROPE (±{rope_binary})")
            plt.axvline(-rope_binary, color="gray", linestyle=":", linewidth=1, label="_nolegend_")
        # HDI 영역
        plt.axvspan(hdi_low, hdi_high, color="orange", alpha=0.3, label=f"95% HDI [{hdi_low:.4f}, {hdi_high:.4f}]")
        plt.xlabel("Uplift in conversion rate")
        plt.ylabel("Density")
        plt.title(f"Posterior uplift with HDI: P(B > A) = {prob_positive:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.show()
        # </editor-fold>

    if stop == True: break
    
# </editor-fold>


stop = False
winner = None
# <editor-fold desc="continuous logic">
for t in range(len(df_continuous)):
    row = df_continuous.iloc[t]
    # A
    n_a = row['count_c']
    mean_a = row['mean_c']
    var_a = row['variance_c']
    # B
    n_b = row['count_t']
    mean_b = row['mean_t']
    var_b = row['variance_t']
    # Posterior update
    mu_a, kappa_a, alpha_a, beta_a = nig_posterior(n_a, mean_a, var_a, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    mu_b, kappa_b, alpha_b, beta_b = nig_posterior(n_b, mean_b, var_b, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    # Sampling
    mu_a_samples, _ = sample_nig_posterior(mu_a, kappa_a, alpha_a, beta_a, sample_num)
    mu_b_samples, _ = sample_nig_posterior(mu_b, kappa_b, alpha_b, beta_b, sample_num)
    # effect
    delta = mu_b_samples - mu_a_samples
    prob_positive = np.mean(delta > 0)
    prob_b_practical = np.mean(delta > rope_continuous)
    prob_a_practical = np.mean(delta < -rope_continuous)
    hdi_low, hdi_high = hdi_of_samples(delta)

    # stop rule
    if stop_rule == 1:
        if prob_positive >= 0.95 and hdi_low > 0 :
            stop = True
            winner = 'B'
        elif prob_positive <= 0.05 and hdi_high < 0 :
            stop = True
            winner = 'A'
    else:
        if prob_b_practical >= 0.95 and hdi_low > rope_continuous :
            stop = True
            winner = 'B'
        elif prob_a_practical >= 0.95 and hdi_high < -rope_continuous :
            stop = True
            winner = 'A'

    # results
    if (stop == True) or (t == len(df_continuous)-1):
        # <editor-fold desc="result">
        print("<<Continuous Results>>")
        print(f"[trial: {t} | winner: {winner}]")
        print(f"prob_positive=P(B>A): {prob_positive:.6f}")
        print(f"prob_b_practical=P(Δ>+{rope_continuous}): {prob_b_practical:.6f}")
        print(f"prob_a_practical=P(Δ<-{rope_continuous}): {prob_a_practical:.6f}")
        print(f"hdi_low, hdi_high: {hdi_low},{hdi_high}")
        # </editor-fold>
        # <editor-fold desc="plot(post_a vs post_b)">
        plt.figure(figsize=(8, 5))
        lo = min(np.percentile(mu_a_samples, 0.5), np.percentile(mu_b_samples, 0.5))
        hi = max(np.percentile(mu_a_samples, 99.5), np.percentile(mu_b_samples, 99.5))
        x = np.linspace(lo, hi, 500)
        plt.plot(x, gaussian_kde(mu_a_samples)(x), label="A")
        plt.plot(x, gaussian_kde(mu_b_samples)(x), label="B")
        plt.xlabel("Mean")
        plt.ylabel("Density")
        plt.title("Posterior distributions of A and B")
        plt.legend()
        plt.tight_layout()
        plt.show()
        # </editor-fold>
        # <editor-fold desc="plot(delta)">
        plt.figure(figsize=(8, 5))
        lo = np.percentile(delta, 0.5)
        hi = np.percentile(delta, 99.5)
        x = np.linspace(lo, hi, 500)
        plt.plot(x, gaussian_kde(delta)(x), label="Delta")
        # 0 기준선
        plt.axvline(0, color="black", linestyle="--", linewidth=1, label="No effect (Δ = 0)")
        if stop_rule == 2:
            plt.axvline(rope_continuous, color="gray", linestyle=":", linewidth=1, label=f"ROPE (±{rope_continuous})")
            plt.axvline(-rope_continuous, color="gray", linestyle=":", linewidth=1, label="_nolegend_")
        # HDI 영역
        plt.axvspan(hdi_low, hdi_high, color="orange", alpha=0.3, label=f"95% HDI [{hdi_low:.4f}, {hdi_high:.4f}]")
        plt.xlabel("Uplift in Mean")
        plt.ylabel("Density")
        plt.title(f"Posterior uplift with HDI: P(B > A) = {prob_positive:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.show()
        # </editor-fold>

    if stop == True: break
# </editor-fold>