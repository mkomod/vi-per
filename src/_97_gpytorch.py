import math
import time

import gpytorch
import torch

from torch.special import log_ndtr, ndtr
from torch.utils.data import TensorDataset, DataLoader


class LogitLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    has_analytic_marginal = False

    def __init__(self, l_max=12.0):
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
        
        res =  \
            torch.dot(y, M.squeeze()) - \
            torch.sum(
                S / math.sqrt(2 * torch.pi) * torch.exp(- 0.5 * M**2 / V) + \
                M * ndtr(M / S)
            ) - \
            torch.sum(
                (-1.0)**(self.l - 1.0) / self.l * (
                    torch.exp( M * self.l + 0.5 * V * (self.l ** 2) + log_ndtr(-M / S - S * self.l)) + \
                    torch.exp(-M * self.l + 0.5 * V * (self.l ** 2) + log_ndtr( M / S - S * self.l))
                )
            )

        return res


class ApproxGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, variational_distribution=None, variational_strategy=None, 
            mean_module=None, covar_module=None):

        if variational_distribution is None:
            # variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
            variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(0))
        
        if variational_strategy is None:
            variational_strat = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            )

        super(ApproxGPModel, self).__init__(variational_strat)

        if mean_module is None:
            # self.mean_module = gpytorch.means.ConstantMean()
            self.mean_module = gpytorch.means.LinearMean(inducing_points.size(1))
        else:
            self.mean_module = mean_module
        
        if covar_module is None:
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(inducing_points.size(1)))
        else:
            self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LogisticGPVI():
    def __init__(self, y, X, likelihood=None, model=None, l_max=12.0, n_inducing=100, 
            n_iter=250, lr=0.1, thresh=1e-4, num_likelihood_samples=1000, 
            seed=1, verbose=True, use_loader=False, batch_size=2048):
        # data
        self.X = X
        self.n = self.X.size()[0]
        self.p = self.X.size()[1]
        self.y = y
        self.dataset = TensorDataset(X, y)

        # data loader
        self.batch_size = batch_size if use_loader else len(self.dataset)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=use_loader, num_workers=6)

        # optimization parameters
        self.n_iter = n_iter
        self.thresh = thresh
        self.lr = lr
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
            self.model = ApproxGPModel(self.inducing_points)
        else:
            self.model = model


    def fit(self, verbose=True):
        start_time = time.time()

        torch.manual_seed(self.seed)
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            [{ 'params': self.model.parameters() }, 
             { 'params': self.likelihood.parameters() }], 
             lr=self.lr
        )

        # define the loss and the ELBO, uses the model and likelihood
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.n)

        # optimize
        for i in range(self.n_iter):
            for X_batch, y_batch in self.loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                self.loss.append(loss.item())

            if (i % 10 == 0):
                if self.verbose or verbose:
                    with torch.no_grad():
                        ll = -mll(self.model(self.X), self.y)
                        print(f"Iter {i}/{self.n_iter} - Loss: {ll.item():.3f}")

            if (i > 2) and ((abs(self.loss[-1] - self.loss[-2]) / self.loss[-2]) < self.thresh):
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
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.n)
            output = self.model(self.X)
            
            return -mll(output, self.y)


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

        