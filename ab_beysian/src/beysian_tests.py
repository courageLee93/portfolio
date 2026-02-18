import numpy as np
from scipy.stats import beta, invgamma, norm

def beta_posterior(success, fail, alpha0=1, beta0=1):
    a = alpha0 + success
    b = beta0 + fail
    return beta(a, b)

def sample_beta_posterior(dist, n=50000):
    return dist.rvs(n)

def nig_posterior(n, mean, var, mu0=0.0, kappa0=0.001, alpha0=1.0, beta0=1.0):
    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * mean) / kappa_n
    alpha_n = alpha0 + n / 2.0
    # var가 표본분산(ddof=1)이면 SSE = (n-1)*var
    # var가 모집단분산(ddof=0)이면 SSE = n*var
    ss = (n - 1.0) * var
    mean_shift = (kappa0 * n * (mean - mu0) ** 2) / kappa_n
    beta_n = beta0 + 0.5 * (ss + mean_shift)
    return mu_n, kappa_n, alpha_n, beta_n

def sample_nig_posterior(mu_n, kappa_n, alpha_n, beta_n, size=50000):
    # 1) σ² ~ InvGamma(α_n, β_n)
    sigma2_samples = invgamma(a=alpha_n, scale=beta_n).rvs(size)
    # 2) μ | σ² ~ Normal(μ_n, σ²/κ_n)
    # mu_samples = norm(loc=mu_n, scale=np.sqrt(sigma2_samples / kappa_n)).rvs(size)
    mu_samples = norm.rvs(loc=mu_n,scale=np.sqrt(sigma2_samples / kappa_n))
    # A/B 비교에서 “평균 차이 Δ = μ_B − μ_A” 이므로, mu_samples만 사용
    return mu_samples, sigma2_samples

def hdi_of_samples(samples, cred_mass = 0.95):
    samples = np.asarray(samples)
    sorted_samp = np.sort(samples)
    n = len(sorted_samp)  # 전체 샘플 개수
    m = int(np.floor(cred_mass * n))  # 포함할 샘플 개수
    # 정렬된 샘플 중에서 “95%를 포함하는 연속된 구간의 길이”를 샘플 개수 기준으로 계산
    # 각 구간의 길이 계산
    # interval_width = sorted_samp[m:] - sorted_samp[:n - m]
    interval_width = []
    for i in range(n - m):
        width = sorted_samp[i + m] - sorted_samp[i]
        interval_width.append(width)
    # 가장 짧은 구간 찾기
    min_idx = np.argmin(interval_width)
    hdi_min = sorted_samp[min_idx]
    hdi_max = sorted_samp[min_idx + m]
    return hdi_min, hdi_max


