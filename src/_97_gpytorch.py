import math
import time
import tqdm

import gpytorch
import torch

from torch.special import log_ndtr, ndtr
from torch.utils.data import TensorDataset, DataLoader
import torch.distributions as dist


class LogitLikelihoodMC(gpytorch.likelihoods._OneDimensionalLikelihood):
    has_analytic_marginal = False

    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        return super().__init__()

    def forward(self, function_samples, *args, **kwargs):
        """ defines the liklihood function """
        output_probs = torch.sigmoid(function_samples)
        return torch.distributions.Bernoulli(probs=output_probs)

    def expected_log_prob(self, y, function_dist, *args, **kwargs):
        """ compute the expected log probability """
        M = function_dist.mean.view(-1, 1)
        S = function_dist.stddev.view(-1, 1)

        norm = dist.Normal(torch.zeros_like(M), torch.ones_like(S))
        samp = norm.sample((self.n_samples, ))
        samp = M + S * samp

        res =  torch.dot(y, M.squeeze()) - \
            torch.sum(torch.mean(torch.log1p(torch.exp(samp)), 0))

        return res
 

class LogitLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    has_analytic_marginal = False

    def __init__(self, l_max=12.0, ):
        self.l_max = l_max
        self.l = torch.arange(1.0, self.l_max*2, 1.0, requires_grad=False)
        return super().__init__()
 
    def forward(self, function_samples, *args, **kwargs):
        """ defines the liklihood function """
        output_probs = torch.sigmoid(function_samples)
        return torch.distributions.Bernoulli(probs=output_probs)

    @torch.jit.export
    def expected_log_prob(self, y, function_dist, *args, **kwargs):
        """ compute the expected log probability """
        M = function_dist.mean.view(-1, 1)
        S = function_dist.stddev.view(-1, 1)
        V = S**2

        M_S = M / S
        ML = M * self.l
        SL = S * self.l
        VL = 0.5 * V * (self.l ** 2)
        
        y_M = torch.dot(y, M.squeeze())
        normal_term = torch.sum(S / math.sqrt(2 * torch.pi) * torch.exp(-0.5 * M**2 / V) + M * ndtr(M_S))
        series_term = torch.sum(
            (-1.0)**(self.l - 1.0) / self.l * (
                torch.exp(ML + VL + log_ndtr(-M_S - SL)) + torch.exp(-ML + VL + log_ndtr(M_S - SL))
            )
        )

        return y_M - normal_term - series_term


class PGLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    # contribution to Eqn (10) in Reference [1].
    def expected_log_prob(self, target, input, *args, **kwargs):
        mean, variance = input.mean, input.variance
        # Compute the expectation E[f_i^2]
        raw_second_moment = variance + mean.pow(2)

        # Translate targets to be -1, 1
        target = target.to(mean.dtype).mul(2.).sub(1.)

        # We detach the following variable since we do not want
        # to differentiate through the closed-form PG update.
        c = raw_second_moment.detach().sqrt()
        # Compute mean of PG auxiliary variable omega: 0.5 * Expectation[omega]
        # See Eqn (11) and Appendix A2 and A3 in Reference [1] for details.
        half_omega = 0.25 * torch.tanh(0.5 * c) / c

        # Expected log likelihood
        res = 0.5 * target * mean - half_omega * raw_second_moment
        # Sum over data points in mini-batch
        res = res.sum(dim=-1)

        return res

    # define the likelihood
    def forward(self, function_samples):
        return torch.distributions.Bernoulli(logits=function_samples)


class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0))
        # variational_distribution = gpytorch.variational.TrilNaturalVariationalDistribution(inducing_points.size(0))
        
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.LinearMean(inducing_points.size(1))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(inducing_points.size(1)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LogisticGPVI():
    def __init__(self, y, X, likelihood=None, model=None, n_inducing=30, n_iter=100, 
        lr=0.1, thresh=1e-4, l_scale=0.5, l_max=12.0, num_likelihood_samples=1000, seed=1, 
        verbose=True, use_loader=False, batches=100, num_workers=0, persistent_workers=False):
        # data
        self.X = X
        self.n = self.X.size()[0]
        self.p = self.X.size()[1]
        self.y = y
        self.dataset = TensorDataset(X, y)

        # data loader
        self.batch_size = self.n // batches if use_loader else len(self.dataset)
        if use_loader:
            self.loader = DataLoader(self.dataset, batch_size=self.batch_size, 
                shuffle=use_loader, drop_last=use_loader, pin_memory=True, 
                num_workers=num_workers, persistent_workers=persistent_workers)
        else:
            self.loader = DataLoader(self.dataset, batch_size=self.batch_size, 
                shuffle=False, drop_last=False, pin_memory=True, 
                num_workers=0)

        # optimization parameters
        self.n_iter = n_iter
        self.thresh = thresh
        self.lr = lr
        self.l_scale = l_scale
        self.loss = []

        # general parameters
        self.seed = seed
        self.verbose = verbose
        self.runtime = 0
        self.num_likelihood_samples = num_likelihood_samples

        # likelihood
        if likelihood is None:
            self.l_max = l_max
            self.likelihood = LogitLikelihood(l_max=self.l_max)
        else:
            self.likelihood = likelihood

        # model 
        if model is None:
            self.n_inducing = n_inducing
            self.inducing_points = torch.randn(self.n_inducing, self.p)
            self.model = GPModel(inducing_points=self.inducing_points)
            self.model.covar_module.base_kernel.lengthscale = torch.ones(self.p) * self.l_scale
        else:
            self.model = model


    def fit(self, verbose=True):
        start_time = time.time()

        torch.manual_seed(self.seed)
        self.model.train()
        self.likelihood.train()

        variational_ngd_optimizer = gpytorch.optim.NGD(self.model.variational_parameters(), num_data=self.y.size(0), lr=self.lr)

        hyperparameter_optimizer = torch.optim.Adam(
            [{ 'params': self.model.hyperparameters() }, 
             { 'params': self.likelihood.parameters() }], 
             lr=self.lr
        )

        # define the loss and the ELBO, uses the model and likelihood
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.n)

        # optimize
        epoch_iter = tqdm.tqdm(range(self.n_iter), disable=not self.verbose)
        ll = [] 
        for i in epoch_iter:
            batch_iter = tqdm.tqdm(self.loader, disable=not self.verbose)

            for X_batch, y_batch in batch_iter:
                variational_ngd_optimizer.zero_grad()
                hyperparameter_optimizer.zero_grad()
                output = self.model(X_batch)
                loss = -mll(output, y_batch)

                epoch_iter.set_postfix(loss=loss.item())
                self.loss.append(loss.item())

                loss.backward()
                variational_ngd_optimizer.step()
                hyperparameter_optimizer.step()

            with torch.no_grad():
                l = -mll(self.model(self.X), self.y) 
                ll.append(l.item())

                if i > 1 and ((abs(ll[-1] - ll[-2]) / abs(ll[-2])) < self.thresh):
                    break

        self.runtime = time.time() - start_time 

        # put the model and likelihood in eval mode
        self.model.eval()
        self.likelihood.eval()


    def predict(self, X):
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), \
            gpytorch.settings.fast_pred_var(), \
            gpytorch.settings.num_likelihood_samples(self.num_likelihood_samples):

            preds = self.likelihood(self.model(X))
            if preds.probs.dim() == 1:
                return preds.probs
            return preds.probs.mean(dim=0)


    def ELBO(self):
        """ Evaluate the evidence lower bound, we want to maximize this """
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.n)
            output = self.model(self.X)
            
            return mll(output, self.y)

    def log_marginal(self, X=None, y=None):
        """ Compute the negative log marginal likelihood, want to minimize this"""
        if X is None or y is None:
            X = self.X
            y = self.y
            
        self.model.eval()
        self.likelihood.eval()
        
        lml = self.likelihood.log_marginal(y, self.model(X))
        return - torch.sum(lml)
    
     
    def ELB0_MC(self, X=None, y=None, n_samples=1000):
        """ compute the ELBO with monte carlo, want to maximize this"""
        if X is None or y is None:
            X = self.X
            y = self.y
            
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), \
            gpytorch.settings.fast_pred_var(), \
            gpytorch.settings.num_likelihood_samples(self.num_likelihood_samples):

            f = self.model(X)

            M = f.mean.view(-1, 1)
            S = f.stddev.view(-1, 1)

            norm = dist.Normal(torch.zeros_like(M), torch.ones_like(S))
            samp = norm.sample((n_samples, ))
            samp = M + S * samp

            res =  torch.dot(y, M.squeeze()) - \
                torch.sum(torch.mean(torch.log1p(torch.exp(samp)), 0))

            return -res - self.model.variational_strategy.kl_divergence()
     

    def neg_log_likelihood(self, X=None, y=None):
        """ compute the negative log likelihood, want to minimize this"""
        if X is None or y is None:
            X = self.X
            y = self.y

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), \
            gpytorch.settings.fast_pred_var(), \
            gpytorch.settings.num_likelihood_samples(self.num_likelihood_samples):

            preds = self.likelihood(self.model(X))
            p = preds.probs if preds.probs.dim() == 1 else preds.probs.mean(dim=0)

            # ensure that p is not 0 or 1 to avoid nan
            p[p == 0] = 1e-7
            p[p == 1] = 1 - 1e-7

            return - torch.sum(y * torch.log(p) + (1 - y) * torch.log(1 - p))


    def sample(self, X, n_samples=1000):
        self.model.eval()

        with torch.no_grad():
            d = self.model(X)
            return d.sample(torch.Size([n_samples])).squeeze().t()
    

    def credible_intervals(self, X, alpha=0.05):
        self.model.eval()

        with torch.no_grad():
            d = self.model(X)
            return d.confidence_region()

        