from src.uplift_evaluate import *
import json

ROOT_DIR = Path(__file__).resolve().parent.parent

uplift_result = data = pd.read_parquet(ROOT_DIR / "uplift" / "result" / "conversion" / "uplift.parquet")
save_path = ROOT_DIR / "uplift" / "report" / "conversion" / "uplift"
save_path.mkdir(parents=True, exist_ok=True)

# treatment / control
treatment = uplift_result["treatment"]
# conversion / non-conversion
outcome = uplift_result["conversion"]
# expected uplift score
uplift = uplift_result["uplift_score"]
# Uplift@k%
k_list = [0.05, 0.1, 0.2, 0.3]
# Random uplift score
rng = np.random.default_rng(42)
random_score = rng.permutation(len(outcome))
random_score = pd.Series(random_score, index=outcome.index)

uplift_summary = uplift_summary_all(uplift = uplift, outcome = outcome, treatment = treatment)
uplift_summary_k = uplift_summary_at_k(uplift = uplift, outcome = outcome, treatment = treatment, k_list = k_list, save_path=save_path)
random_summary = uplift_summary_all(uplift = random_score, outcome = outcome, treatment = treatment)
random_summary_k = uplift_summary_at_k(uplift = random_score, outcome = outcome, treatment = treatment, k_list = k_list, save_path=save_path)
report = {
    "uplift": {
        "summary_all": uplift_summary,
        "summary_at_k": uplift_summary_k.to_dict(orient="records")
    },
    "random":{
        "summary_all": random_summary,
        "summary_at_k": random_summary_k.to_dict(orient="records")
    }
}
with open(save_path / "report.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

plt_qini_curve(uplift = uplift, outcome = outcome, treatment = treatment,save_path=save_path)
plt_uplift_curve(uplift = uplift, outcome = outcome, treatment = treatment,save_path=save_path)
for k in k_list:
    plt_uplift_curve_k(uplift=uplift, outcome=outcome, treatment=treatment, k_frac=k, save_path=save_path)
plt_treat_control_rates_by_decile(uplift = uplift, outcome = outcome, treatment = treatment,save_path=save_path)
plt_treat_control_rates_by_k(uplift = uplift, outcome = outcome, treatment = treatment, k_list = k_list, save_path=save_path)



