import math
import torch

from lib.poisson_renewal import ModelPoissonRenewal, Learner


def wasserstein_distance(t1, t2):
    n = min(len(t1), len(t2))
    m = max(len(t1), len(t2))
    if len(t1) == 0:
        T = max(t2)
    elif len(t2) == 0:
        T = max(t1)
    else:
        T = max(max(t1), max(t2))
    val = sum(abs(t1[:n] - t2[:n])) + (m - n) * T - sum(t1[n:]) - sum(t2[n:])
    return val


def estimate_reproduction_number(daily_count_arr, beta, T):
    model = ModelPoissonRenewal()
    learner = Learner(model, lr=0.01, lr_gamma=1.0, tol=1e-6, max_iter=10000)
    log_r_init = torch.tensor([0.0], dtype=torch.float64)
    log_r_hat = learner.fit(log_r_init, daily_count_arr, beta, T)
    log_r_hat = log_r_hat.detach().numpy()[0]
    r_hat = math.exp(log_r_hat)
    return r_hat
