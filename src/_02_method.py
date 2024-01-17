import math
import time

import torch
import torch.distributions as dist

import hamiltorch

from torch.special import log_ndtr, ndtr
from torch.nn.functional import logsigmoid


def KL_mvn(m, S, mu, Sig):
    """
    Can also be computed via:
        mvn1 = dist.MultivariateNormal(m, S)
        mvn2 = dist.MultivariateNormal(mu, Sig)
        dist.kl.kl_divergence(mvn1, mvn2)
    """
    p = m.size()[0]
    res = 0.5 * (torch.logdet(Sig) - torch.logdet(S) -  p + 
                 torch.trace(Sig.inverse() @ S) + 
                 (mu - m).t() @ Sig.inverse() @ (mu - m))
    return res


def KL(m, s, mu, sig):
    """
    Compute the KL divergence between two Gaussians
    :param m: mean of variational distribution
    :param s: standard deviation of variational distribution
    :param mu: mean of prior
    :parma sig: standard deviation of prior
    :return: KL divergence
    """
    res = torch.log(sig / s) + 0.5 * ((s ** 2 + (m - mu) ** 2) / sig ** 2 - 1)
    return torch.sum(res)


def KL_MC(m, s, mu, sig):
    """ 
    Compute the KL divergence between two Gaussians with monte carlo
    :param m: mean of variational distribution
    :param s: standard deviation of variational distribution
    :param mu: mean of prior
    :parma sig: standard deviation of prior
    :return: KL divergence
    """
    d1 = dist.Normal(m, s)
    d2 = dist.Normal(mu, sig) 

    x = d1.sample((1000,))
    return torch.mean(torch.sum(d1.log_prob(x) - d2.log_prob(x), 1))
 

def ELL_TB(m, s, y, X, l_max = 10.0, XX=None):
    """
    Compute the expected negative log-likelihood
    :return: ELL
    """
    M = X @ m

    if XX is None:
        S = torch.sum(X ** 2 * s ** 2, dim=1)
    else:
        S = torch.sum(XX * s ** 2, dim=1)
    
    S = torch.sqrt(S)

    l = torch.arange(1.0, l_max*2, 1.0, requires_grad=False, dtype=torch.float64)

    M = M.unsqueeze(1)
    S = S.unsqueeze(1)
    l = l.unsqueeze(0)


    res =  \
        torch.dot(- y, X @ m) + \
        torch.sum(
            S / math.sqrt(2 * torch.pi) * torch.exp(- 0.5 * M**2 / S**2) + \
            M * ndtr(M / S)
        ) + \
        torch.sum(
            (-1.0)**(l - 1.0) / l * (
                torch.exp( M @ l + 0.5 * S**2 @ (l ** 2) + log_ndtr(-M / S - S @ l)) + \
                torch.exp(-M @ l + 0.5 * S**2 @ (l ** 2) + log_ndtr( M / S - S @ l))
            )
        )

    return res



def ELL_TB_mvn(m, S, y, X, l_max = 10.0):
    """
    Compute the expected negative log-likelihood
    :return: ELL
    """
    M = X @ m
    # S = torch.diag(X @ S @ X.t()) # this is too slow!
    
    try: 
        U = torch.linalg.cholesky(S)
        S = torch.sum((X @ U) ** 2, dim=1)
    except:
        S = torch.sum(X * (S @ X.t()).t(), dim=1)
    
    S = torch.sqrt(S)

    l = torch.arange(1.0, l_max*2, 1.0, requires_grad=False, dtype=torch.float64)

    M = M.unsqueeze(1)
    S = S.unsqueeze(1)
    l = l.unsqueeze(0)

    res =  \
        torch.dot(- y, M.squeeze()) + \
        torch.sum(
            S / math.sqrt(2 * torch.pi) * torch.exp(- 0.5 * M**2 / S**2) + \
            M * ndtr(M / S)
        ) + \
        torch.sum(
            (-1.0)**(l - 1.0) / l * (
                torch.exp( M @ l + 0.5 * S**2 @ (l ** 2) + log_ndtr(-M / S - S @ l)) + \
                torch.exp(-M @ l + 0.5 * S**2 @ (l ** 2) + log_ndtr( M / S - S @ l))
            )
        )

    return res


def ELL_MC(m, s, y, X, n_samples=1000):
    """
    Compute the expected negative log-likelihood with monte carlo
    :return: ELL
    """
    M = X @ m
    S = torch.sqrt(X ** 2 @ s ** 2)

    norm = dist.Normal(torch.zeros_like(M), torch.ones_like(S))
    samp = norm.sample((n_samples, ))
    samp = M + S * samp

    res =  torch.dot( - y, M) + \
        torch.sum(torch.mean(torch.log1p(torch.exp(samp)), 0))

    return res


def ELL_MC_mvn(m, S, y, X, n_samples=1000):
    """
    Compute the expected negative log-likelihood with monte carlo
    :return: ELL
    """
    M = X @ m
    # S = torch.diag(X @ S @ X.t())

    try: 
        U = torch.linalg.cholesky(S)
        S = torch.sum((X @ U) ** 2, dim=1)
    except:
        S = torch.sum(X * (S @ X.t()).t(), dim=1)
    
    S = torch.sqrt(S)

    norm = dist.Normal(torch.zeros_like(M), torch.ones_like(S))
    samp = norm.sample((n_samples, ))
    samp = M + S * samp

    res =  torch.dot( - y, M) + \
        torch.sum(torch.mean(torch.log1p(torch.exp(samp)), 0))

    return res
 

def ELL_Jak(m, s, t, y, X):
    """
    Compute the expected negative log-likelihood using the bound introduced
    by Jaakkola and Jordan (2000)
    :return: ELL
    """
    M = X @ m
    a_t = (torch.sigmoid(t) - 0.5) / t
    S = torch.diag(s**2) + torch.outer(m, m)

    try:
        U = torch.linalg.cholesky(S)
        B = a_t * torch.sum((X @ U) ** 2, dim=1)
    except:
        B = a_t * torch.sum(X * (S @ X.t()).t(), dim=1)

    res = - torch.dot(y, M) - torch.sum(logsigmoid(t)) + \
        0.5 * torch.sum(M + t) + 0.5 * torch.sum(B)   - \
        0.5 * torch.sum(a_t * t ** 2)

    return res


def ELL_Jak_mvn(m, S, t, y, X):
    """
    Compute the expected negative log-likelihood using the bound introduced
    by Jaakkola and Jordan (2000)
    :return: ELL
    """
    M = X @ m
    a_t = (torch.sigmoid(t) - 0.5) / t
    SS = S + torch.outer(m, m)

    try:
        U = torch.linalg.cholesky(SS)
        B = a_t * torch.sum((X @ U) ** 2, dim=1)
    except:
        B = a_t * torch.sum(X * (SS @ X.t()).t(), dim=1)

    res = - torch.dot(y, M) - torch.sum(logsigmoid(t)) + \
        0.5 * torch.sum(M + t) + 0.5 * torch.sum(B)   - \
        0.5 * torch.sum(a_t * t ** 2)

    return res


def ELBO_TB(m, u, y, X, mu, sig, l_max = 10.0, XX=None):
    """
    Compute the negative of the ELBO
    :return: ELBO
    """
    s = torch.exp(u)
    return ELL_TB(m, s, y, X, l_max=l_max, XX=XX) + KL(m, s, mu, sig)


def ELBO_TB_mvn(m, u, y, X, mu, Sig, l_max = 10.0):
    """
    Compute the negative of the ELBO
    :return: ELBO
    """
    p = Sig.size()[0]
    L = torch.zeros(p, p, dtype=torch.double)
    L[torch.tril_indices(p, p, 0).tolist()] = u
    S = L.t() @ L
    
    return ELL_TB_mvn(m, S, y, X, l_max=l_max) + KL_mvn(m, S, mu, Sig)


def ELBO_MC(m, u, y, X, mu, sig, n_samples=1000):
    """
    Compute the negative of the ELBO
    :return: ELBO
    """
    s = torch.exp(u)
    return ELL_MC(m, s, y, X, n_samples) + KL(m, s, mu, sig)


def ELBO_MC_mvn(m, u, y, X, mu, Sig, n_samples=1000):
    """
    Compute the negative of the ELBO
    :return: ELBO
    """
    p = Sig.size()[0]
    L = torch.zeros(p, p, dtype=torch.double)
    L[torch.tril_indices(p, p, 0).tolist()] = u
    S = L.t() @ L

    return ELL_MC_mvn(m, S, y, X, n_samples) + KL_mvn(m, S, mu, Sig)


def ELBO_Jak(m, s, t, y, X, mu, sig):
    """
    Compute the negative of the ELBO using the bound introduced by
    Jaakkola and Jordan (2000)
    :return: ELBO
    """
    return ELL_Jak(m, s, t, y, X) + KL(m, s, mu, sig)


def ELBO_Jak_mvn(m, S, t, y, X, mu, Sig, cov=None):
    """
    Compute the negative of the ELBO using the bound introduced by
    Jaakkola and Jordan (2000)
    :return: ELBO
    """
    return ELL_Jak_mvn(m, S, t, y, X) + KL_mvn(m, S, mu, Sig)


class LogisticVI:
    def __init__(self, dat, intercept=False, method=0, 
        mu=None, sig=None, Sig=None, m_init=None, s_init=None,
        n_iter=1200, thresh=1e-8, verbose=False, lr=0.08,
        l_max=12.0, adaptive_l=False, l_thresh=1e-2, 
        n_samples=500, seed=1):
        """ 
        Initialize the class
        :param dat: data
        :param mu: mean of prior
        :param sig: standard deviation of prior
        :param mu_init: mean of variational distribution
        :param s_init: standard deviation of variational distribution
        :param method: method to use
        :param seed: seed for reproducibility
        """
        torch.manual_seed(seed)

        self.X = dat["X"]
        self.intercept = intercept
        self.l_max = l_max
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.runtime = 0
        self.seed = seed
        self.thresh = thresh
        self.verbose = verbose
        self.adaptive_l = adaptive_l
        self.l_thresh = l_thresh
        self.lr = lr
        self.y = dat["y"]

        if adaptive_l:
            self.l_terms = float(int(l_max / 2))
        else:
            self.l_terms = l_max

        if intercept:
            self.X = torch.cat((torch.ones(self.X.size()[0], 1), self.X), 1)

        self.n = self.X.size()[0]
        self.p = self.X.size()[1]

        self.mu = torch.zeros(self.p, dtype=torch.double) if mu is None else mu
        self.sig = torch.ones(self.p, dtype=torch.double) if sig is None else sig
        self.Sig = torch.eye(self.p, dtype=torch.double) if Sig is None else Sig

        self.m_init = torch.randn(self.p, dtype=torch.double) if m_init is None else m_init

        if s_init is None:
            self.u_init = torch.tensor([-1.] * self.p, dtype=torch.double)
            self.s_init = torch.exp(self.u_init)
        else:
            self.s_init = s_init
            self.u_init = torch.log(s_init)

        self.m = self.m_init.clone()
        self.u = self.u_init.clone()
        self.method = method
        self.loss = []          

    
    def fit(self):
        """
        Fit the model
        """
        # fit the method with the defaults
        self._fit()

        # if there is an issue i.e. nan values, reduce the learning rate 
        while (torch.any(self.m.isnan()) or torch.any(self.u.isnan())):
            self.lr = max(0.01, self.lr - 0.01)
            self._fit()
            

    def _fit(self):
        """ fitting method """ 
        start = time.time()

        # reset parameters
        # self.m = self.m_init.clone()
        # self.u = self.u_init.clone()

        self.m.requires_grad = True
        self.u.requires_grad = True

        self.loss = []

        optimizer = torch.optim.Adam([self.m, self.u], lr=self.lr)

        if self.method == 0:
            if self.verbose:
                print("Fitting with proposed bound, diagonal covariance variational family")
            
            self.XX = self.X ** 2
            self._trainig_loop(optimizer)
            self.s = torch.exp(self.u)

        elif self.method == 1:
            if self.verbose:
                print("Fitting with proposed bound, full covariance variational family")

            self.u = torch.ones(int(self.p * (1 + self.p) / 2.0), dtype=torch.double)
            self.u = self.u * 1/self.p
            self.u_init = self.u.clone()
            self.u.requires_grad = True

            optimizer = torch.optim.Adam([self.m, self.u], lr=self.lr)
            self._trainig_loop(optimizer)

            L = torch.zeros_like(self.Sig)
            L[torch.tril_indices(self.p, self.p, 0).tolist()] = self.u
            self.S = L.t() @ L
            self.s = torch.sqrt(torch.diag(self.S))

        elif self.method == 2:
            if self.verbose:
                print("Fitting with Jaakkola and Jordan bound, diagonal covariance variational family")

            self._fit_Jak()

        elif self.method == 3:
            if self.verbose:
                print("Fitting with Jaakkola and Jordan bound, full covariance variational family")

            self._fit_Jak_mvn()
            self.s = torch.sqrt(torch.diag(self.S))
            U = torch.linalg.cholesky(self.S)
            self.u = U[torch.tril_indices(self.p, self.p, 0).tolist()]

        elif self.method == 4:
            if self.verbose:
                print("Fitting with Monte Carlo, diagonal covariance variational family")
            
            self._trainig_loop(optimizer) 
            self.s = torch.exp(self.u)

        elif self.method == 5:
            if self.verbose:
                print("Fitting with Monte Carlo, full covariance variational family")

            self.u = torch.ones(int(self.p * (1 + self.p) / 2.0), dtype=torch.double)
            self.u = self.u * 1.0/self.p
            self.u_init = self.u.clone()
            self.u.requires_grad = True
            
            optimizer = torch.optim.Adam([self.m, self.u], lr=self.lr)
            self._trainig_loop(optimizer)

            L = torch.zeros_like(self.Sig)
            L[torch.tril_indices(self.p, self.p, 0).tolist()] = self.u
            self.S = L.t() @ L
            self.s = torch.sqrt(torch.diag(self.S))

        else:
            raise ValueError("Method not recognized")

        self.runtime = time.time() - start 
        self.l_terms = self.l_max

    
    def _loss_below_thresh(self, thresh):
        """
        Return True if the relative loss is below the threshold
        """
        relative_loss = abs(self.loss[-1] - self.loss[-2]) / self.loss[-2]
        return relative_loss < self.thresh


    def _trainig_loop(self, optimizer):
        """
        Training loop
        """
        for epoch in range(self.n_iter):
            optimizer.zero_grad()
            l = self.ELBO()
            self.loss.append(l.item())

            l.backward()
            optimizer.step()

            if torch.any(self.m.isnan()) or torch.any(self.u.isnan()):
                break

            if self.adaptive_l and (self.method == 0 or self.method == 1): 
                if epoch > 2 and \
                    self._loss_below_thresh(self.l_thresh) and \
                    self.l_terms < self.l_max:
                        self.l_terms += 1.0


            if epoch > 2 and self._loss_below_thresh(self.thresh):
                break

            if epoch % 20 == 0 and self.verbose:
                print(epoch, l.item())


    def ELBO(self):
        if self.method == 0:
            return ELBO_TB(self.m, self.u, self.y, self.X, self.mu, self.sig, self.l_terms, XX=self.XX)
        elif self.method == 1:
            return ELBO_TB_mvn(self.m, self.u, self.y, self.X, self.mu, self.Sig, self.l_terms)
        elif self.method == 2:
            return ELBO_Jak(self.m, self.s, self.t, self.y, self.X, self.mu, self.sig)
        elif self.method == 3:
            return ELBO_Jak_mvn(self.m, self.S, self.t, self.y, self.X, self.mu, self.Sig)
        elif self.method == 4:
            return ELBO_MC(self.m, self.u, self.y, self.X, self.mu, self.sig, self.n_samples)
        elif self.method == 5:
            return ELBO_MC_mvn(self.m, self.u, self.y, self.X, self.mu, self.Sig, self.n_samples)
        else:
            raise ValueError("Method not recognized")

    def _ELBO_MC(self, n_samples=10000):
        """ this is the normal elbo, we want to maximize this"""
        with torch.no_grad():
            if self.method == 0 or self.method == 2 or self.method == 4:
                return - ELBO_MC(self.m, self.u, self.y, self.X, self.mu, self.sig, n_samples=n_samples)
            elif self.method == 1 or self.method == 3 or self.method == 5:
                return - ELBO_MC_mvn(self.m, self.u, self.y, self.X, self.mu, self.Sig, n_samples=n_samples)
            else:
                raise ValueError("Method not recognized")

    def neg_log_likelihood(self, n_samples=1000):
        if self.intercept:
            X = torch.cat((torch.ones(X.size()[0], 1), X), 1)

        M = self.sample(n_samples=n_samples)
        p = torch.mean(torch.sigmoid(X @ M), 0)

        p[p == 0] = 1e-7
        p[p == 1] = 1 - 1e-7

        return -torch.sum(self.y * torch.log(p) + (1 - self.y) * torch.log(1 - p))

    def _fit_Jak(self):
        """
        Fit the model using Jaak bound, this is analytic
        """
        self.t = torch.ones(self.n, dtype=torch.double)
        self.m.requires_grad = False
        self.u.requires_grad = False
        self.s = torch.exp(self.u)
        V = self.X.t() @ (self.y - 0.5)

        for epoch in range(self.n_iter):
            a_t = (torch.sigmoid(self.t) - 0.5) / self.t
            # C = self.X.t() @ torch.diag(a_t) @ self.X 
            C = (self.X.t() * a_t) @ self.X

            self.m = torch.inverse(C + torch.diag(1/self.sig**2)) @ (self.mu / self.sig**2 + V)
            # self.s = torch.sqrt(torch.diag(torch.inverse(torch.diag(1/self.sig**2) + C)))
            self.s = 1 / torch.sqrt(torch.diag(C) + 1 / self.sig**2)

            S = torch.diag(self.s**2)  + torch.outer(self.m, self.m)
            try:
                U = torch.linalg.cholesky(S)
                self.t = torch.sqrt(torch.sum((self.X @ U) ** 2, dim=1))
            except:
                self.t = torch.sqrt(torch.sum(self.X * (S @ self.X.t()).t(), dim=1))
            
            l = ELL_Jak(self.m, self.s, self.t, self.y, self.X) + KL(self.m, self.s, self.mu, self.sig)
            self.loss.append(l.item())

            if epoch > 2 and self._loss_below_thresh(self.thresh):
                break
            
            if epoch % 20 == 0 and self.verbose:
                print(epoch, l.item())


    def _fit_Jak_mvn(self):
        """
        Fit the model using Jaak bound
        """
        self.t = torch.ones(self.n, dtype=torch.double)
        self.m.requires_grad = False
        self.u.requires_grad = False
        self.S = torch.diag(torch.exp(self.u))
        V = self.X.t() @ (self.y - 0.5)

        for epoch in range(self.n_iter):
            a_t = (torch.sigmoid(self.t) - 0.5) / self.t
            # C = self.X.t() @ torch.diag(a_t) @ self.X
            C = (self.X.t() * a_t) @ self.X
            
            self.m = self.S @ (torch.inverse(self.Sig) @ self.mu + V)
            self.S = torch.inverse(torch.inverse(self.Sig) + C)

            S = self.S  + torch.outer(self.m, self.m)

            try:
                U = torch.linalg.cholesky(S)
                self.t = torch.sqrt(torch.sum((self.X @ U) ** 2, dim=1))
            except:
                self.t = torch.sqrt(torch.sum(self.X * (S @ self.X.t()).t(), dim=1))
            
            l = ELL_Jak_mvn(self.m, self.S, self.t, self.y, self.X) + KL_mvn(self.m, self.S, self.mu, self.Sig)
            self.loss.append(l.item())

            if epoch > 2 and self._loss_below_thresh(self.thresh):
                break

            if epoch % 20 == 0 and self.verbose:
                print(epoch, l.item())


    def predict(self, X):
        """
        Predict using the model
        :param X: data
        :return: predictions
        """
        if self.intercept:
            X = torch.cat((torch.ones(X.size()[0], 1), X), 1)

        return torch.sigmoid(X @ self.m)


    def sample(self, n_samples=10000):
        if self.method == 0 or self.method == 2 or self.method == 4:
            mvn = dist.MultivariateNormal(self.m, torch.diag(self.s**2))
        if self.method == 1 or self.method == 3 or self.method == 5:
            mvn = dist.MultivariateNormal(self.m, self.S)

        samp = mvn.sample((n_samples, ))

        return samp


    def credible_intervals(self, width=torch.tensor(0.95)):
        d = dist.Normal(self.m, self.s)
        a = (1 - width) / 2

        lower = d.icdf(a)
        upper = d.icdf(1 - a)

        return torch.stack((lower, upper)).t()
            

class LogisticMCMC:
    def __init__(self, dat, intercept=False, n_iter=3e4, burnin=25e3, 
        mu=None, sig=None, Sig=None, verbose=False, 
        step_size=0.01, L=25, seed=1):
        """ 
        Initialize the class
        :param dat: data
        :param mu: mean of prior
        :param sig: standard deviation of prior
        :param mu_init: mean of variational distribution
        :param s_init: standard deviation of variational distribution
        :param method: method to use
        :param seed: seed for reproducibility
        """
        torch.manual_seed(seed)
        self.seed=1
        self.y = dat["y"]
        self.X = dat["X"]
        self.intercept = intercept
        self.verbose = verbose
        self.time = time
        self.runtime = 0
        self.burnin = int(burnin)
        self.n_iter = int(n_iter)
        self.step_size = step_size
        self.L=L

        if intercept:
            self.X = torch.cat((torch.ones(self.X.size()[0], 1), self.X), 1)

        self.n = self.X.size()[0]
        self.p = self.X.size()[1]
        self.b_init = torch.randn(self.p, dtype=torch.double)
        self.loss = []

        # priors
        self.mu = torch.zeros(self.p, dtype=torch.double) if mu is None else mu
        self.sig = torch.ones(self.p, dtype=torch.double) if sig is None else sig
        self.Sig = torch.eye(self.p, dtype=torch.double) if Sig is None else Sig

        if Sig is not None: 
            self.prior = dist.MultivariateNormal(self.mu, self.Sig)
        else:
            self.prior = dist.Normal(self.mu, self.sig)



    def log_prior(self, b):
        """
        Compute the log-prior
        """
        return torch.sum(self.prior.log_prob(b))


    def log_likelihood(self, b):
        """
        Compute the log-likelihood
        """
        return torch.sum(self.y * (self.X @ b) - torch.log1p(torch.exp(self.X @ b)))

    def log_prob(self, b):
        """
        Compute the log-probability
        """
        return self.log_prior(b) + self.log_likelihood(b)
        
    
    def fit(self):
        """
        Fit the model
        """
        if self.time:
            start = time.time()

        self._fit()

        if self.time: 
            self.runtime = time.time() - start

        self.m = self.B.mean(dim=0)


    def _fit(self):
        """
        Fit the model using MH
        """
        hamiltorch.set_random_seed(self.seed)
        self.B = hamiltorch.sample(log_prob_func=self.log_prob, 
                    params_init=self.b_init, num_samples=self.n_iter, 
                    step_size=self.step_size, num_steps_per_sample=self.L, 
                    burn=self.burnin, verbose=not self.verbose)
        self.B = torch.stack(self.B)


    def predict(self, X):
        """ predict using the posterior mean """
        if self.intercept:
            X = torch.cat((torch.ones(X.size()[0], 1), X), 1)
        
        return torch.sigmoid(X @ self.m)


    def credible_intervals(self, width=0.95):
        a = (1 - width) / 2
        lower = self.B.quantile(a, dim=0)
        upper = self.B.quantile(1-a, dim=0)

        return torch.stack((lower, upper)).t()

