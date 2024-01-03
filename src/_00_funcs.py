import math
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
import dsdl
import gpytorch
import tqdm

from torcheval import metrics as tm
import torch.distributions as dist

from torcheval.metrics import BinaryAUROC
from joblib import Parallel, delayed


from _97_gpytorch import LogisticGPVI, GPModel, LogitLikelihood, PGLikelihood, LogitLikelihoodMC



def evaluate_method(fit, dat, method="vi"):
    auc = tm.BinaryAUROC()
    auc.update(fit.predict(dat["X"]), dat["y"])

    mse = tm.MeanSquaredError()
    creds = fit.credible_intervals()

    if fit.intercept:
        mse.update(fit.m[1:], dat["b"])
        cov = torch.sum(torch.logical_and(creds[1:,0] < dat["b"], dat["b"] < creds[1:,1])) / dat["b"].size()[0]
    else:
        mse.update(fit.m, dat["b"])
        cov = torch.sum(torch.logical_and(creds[:, 0] < dat["b"], dat["b"] < creds[:, 1])) / dat["b"].size()[0]

    cred_size = torch.mean(torch.diff(creds))


    if method == "vi":
        elbo_mc = fit._ELBO_MC().item()
        B = fit.sample(5000)
    else:
        elbo_mc = 0.0
        B = fit.B

    f_true = dat["X"] @ dat["b"].t()
    f_pred = dat["X"] @ B.t()
    
    f_mean = f_pred.mean(1)
    lower, upper = f_pred.quantile(0.025, dim=1), f_pred.quantile(0.975, dim=1)

    f_mse = torch.mean((f_mean - f_true)**2)
    f_cov = torch.sum(torch.logical_and(lower < f_true, f_true < upper)) / f_true.size()[0]
    f_cred_size = torch.mean(torch.diff(torch.cat([lower.unsqueeze(1), upper.unsqueeze(1)], dim=1), dim=1))

    return elbo_mc, auc.compute().item(), \
        mse.compute().item(), cov, cred_size, \
        f_mse.item(), f_cov.item(), f_cred_size.item(), \
        fit.runtime


def process_dataset(dataset_name, standardize=True):
    data = dsdl.load(dataset_name)
    X, y = data.get_train()
    X = X.todense()

    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    # ensure two classes are 0 and 1
    classes = torch.unique(y)
    y[y == classes[0]] = 0
    y[y == classes[1]] = 1

    X_test, y_test = data.get_test()
    

    if X_test is None:
        # take 10% of training data as test data
        idx = torch.ones(X.size()[0], dtype=torch.bool)
        id0 = torch.randperm(X.size()[0])
        id0 = id0[:int(0.1 * X.size()[0])]
        idx[id0] = False

        X_test = X[torch.logical_not(idx)]
        y_test = y[torch.logical_not(idx)]

        X = X[idx]
        y = y[idx]
    else:
        X_test = X_test.todense()
        X_test = torch.tensor(X_test, dtype=torch.float)
        y_test = torch.tensor(y_test, dtype=torch.float)

        y_test[y_test == classes[0]] = 0
        y_test[y_test == classes[1]] = 1


    if standardize:
        Xmean = X.mean(dim=0)
        Xstd = X.std(dim=0)
        X = (X - Xmean) / Xstd
        X_test = (X_test - Xmean) / Xstd


    if X_test.size()[0] > 5000:
        # randomly select 5000 test points
        idx = torch.randperm(X_test.size()[0])
        X_test = X_test[idx]
        y_test = y_test[idx]
    
    return y, X, y_test, X_test


def analyze_dataset(seed, y, X, y_test, X_test, n_iter=200, n_inducing=50, thresh=1e-6,
                 verbose=False, use_loader=False, batches=20, lr=0.05, standardize=True):
    torch.manual_seed(seed)
    print(f"Run: {seed}")
        
    f0 = LogisticGPVI(y, X, n_inducing=n_inducing, n_iter=n_iter, thresh=thresh, verbose=verbose, 
                            use_loader=use_loader, batches=batches, seed=seed, lr=0.07)
    f0.fit()

    
    f1 = LogisticGPVI(y, X, likelihood=LogitLikelihoodMC(1000), n_inducing=n_inducing, n_iter=n_iter, thresh=thresh, 
                            verbose=verbose, use_loader=use_loader, batches=batches, seed=seed, lr=0.03)
    f1.fit()

    f2 = LogisticGPVI(y, X, likelihood=PGLikelihood(), n_inducing=n_inducing, n_iter=n_iter, thresh=thresh, 
                            verbose=verbose, use_loader=use_loader, batches=batches, seed=seed, lr=0.07)
    f2.fit()

    return torch.tensor([
        evaluate_method_application(f0, X, y, X_test, y_test), 
        evaluate_method_application(f1, X, y, X_test, y_test),
        evaluate_method_application(f2, X, y, X_test, y_test)
    ])



def evaluate_method_application(func, X_train, y_train, X_test, y_test):
    y_pred = func.predict(X_test)

    auc = BinaryAUROC()
    auc.update(y_pred, y_test)
    auc_test = auc.compute().item()

    auc = BinaryAUROC()
    auc.update(func.predict(X_train), y_train)
    auc_train = auc.compute().item()

    lower, upper = func.credible_intervals(X_test)
    ci_width = (upper - lower).mean().item()

    return func.runtime, ci_width, \
            auc_test, auc_train, \
            func.neg_log_likelihood().item(), func.neg_log_likelihood(X_test, y_test), \
            func.log_marginal().item(), func.log_marginal(X_test, y_test).item(), \
            func.ELB0_MC().item(), func.ELB0_MC(X_test, y_test).item()


def print_results(res):
    
    for i in range(res.shape[0]):
        mean, sd = res[i].mean(dim=0), res[i].std(dim=0)
        l = ""
        for m, s in zip(mean, sd):
            l += f"{m:.2f} ({s:.2f}) \t "
            
        print(l[:-2])


def sf(x, n):
     return '{:g}'.format(float('{:.{p}g}'.format(x, p=n)))


def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    time_components = []

    if hours:
        time_components.append("{}h".format(int(hours)))
    if minutes:
        time_components.append("{}m".format(int(minutes)))

    time_components.append("{}s".format(sf(seconds, 2)))

    # Join the components and format as a string
    result = " ".join(time_components)

    return result

