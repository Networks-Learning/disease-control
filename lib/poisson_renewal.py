import torch
from collections import defaultdict


class ModelPoissonRenewal:

    def set_data(self, count_arr, beta, T):
        # Counts from 1:end
        self.count = torch.tensor(count_arr, dtype=torch.double)[1:]
        # Cumumaltive sum of counts for T-i:i for each i in 1:end
        self.count_cumsum = torch.tensor([count_arr[max(i-T-1,0):i].sum() for i in range(1, len(count_arr))], dtype=torch.double)
        self.beta = beta
        self.T = len(count_arr)-1 if T is None else T

    def log_likelihood(self, log_r):
        lamb = torch.exp(log_r) * self.beta * self.count_cumsum
        vals = self.count * torch.log(lamb) - torch.lgamma(self.count+1) - lamb
        return vals.sum()

    def objective(self, log_r):
        return -1.0 * self.log_likelihood(log_r)


class Learner:

    def __init__(self, model, lr, lr_gamma, tol, max_iter):
        self.model = model
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.tol = tol
        self.max_iter = max_iter

    def _set_data(self, count_arr, beta, T):
        self.model.set_data(count_arr, beta, T)

    def _check_convergence(self):
        if torch.abs(self.coeffs - self.coeffs_prev).max() < self.tol:
            return True
        return False

    def fit(self, x0, daily_count_arr, beta, T=None, callback=None):
        self._set_data(daily_count_arr, beta, T)
        
        self.coeffs = x0.clone().detach().requires_grad_(True)
        self.coeffs_prev = self.coeffs.detach().clone()
       
        self.optimizer = torch.optim.Adam([self.coeffs], lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.lr_gamma)
        
        for t in range(self.max_iter):
            self._n_iter_done = t
            # Gradient step
            self.optimizer.zero_grad()
            self.loss = self.model.objective(self.coeffs)
            self.loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if torch.isnan(self.coeffs).any():
                raise ValueError('NaNs in coeffs! Stop optimization...')
            if torch.isnan(self.loss).any():
                raise ValueError('NaNs is loss! Stop optimization...')

            # Convergence check
            if self._check_convergence():
                break
            elif callback:  # Callback at each iteration
                callback(self, end='')
            self.coeffs_prev = self.coeffs.detach().clone()
        if callback:  # Callback before the end
            callback(self, end='\n', force=True)
        return self.coeffs


class CallbackMonitor:

    def __init__(self, print_every=10):
        self.print_every = print_every

    def __call__(self, learner_obj, end='', force=False):
        t = learner_obj._n_iter_done + 1
        if force or (t % self.print_every == 0):
            dx = torch.abs(learner_obj.coeffs - learner_obj.coeffs_prev).max()
            print("\r    "
                  f"iter: {t:>4d}/{learner_obj.max_iter:>4d} | "
                  f"R: {learner_obj.coeffs[0]:.4f} | "
                  f"loss: {learner_obj.loss:.4f} | "
                  f"dx: {dx:.2e}"
                  "    ", end=end, flush=True)
