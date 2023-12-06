import sys
import math

import torch
import numpy as np
import matplotlib.pyplot as plt
import dsdl
import gpytorch
import tqdm

from torcheval.metrics import BinaryAUROC
from joblib import Parallel, delayed

sys.path.append("..")

from src._97_gpytorch import LogisticGPVI, GPModel, LogitLikelihood, PGLikelihood



def process_dataset(dataset_name, standardize=True):
    data = dsdl.load(dataset_name)
    X, y = data.get_train()
    X = X.todense()

    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    if standardize:
        Xmean = X.mean(dim=0)
        Xstd = X.std(dim=0)
        X = (X - Xmean) / Xstd

    # ensure two classes are 0 and 1
    classes = torch.unique(y)
    y[y == classes[0]] = 0
    y[y == classes[1]] = 1

    X_test, y_test = data.get_test()
    
    if X_test is None:
        X_test = X
        y_test = y
    else:
        X_test = X_test.todense()
        X_test = torch.tensor(X_test, dtype=torch.float)
        y_test = torch.tensor(y_test, dtype=torch.float)

        if standardize:
            X_test = (X_test - Xmean) / Xstd

        y_test[y_test == classes[0]] = 0
        y_test[y_test == classes[1]] = 1

    if X_test.size()[0] > 5000:
        # randomly select 5000 test points
        idx = torch.randperm(X_test.size()[0])
        X_test = X_test[idx]
        y_test = y_test[idx]
    
    return y, X, y_test, X_test


def analyze_dataset(seed, y, X, y_test, X_test, n_iter=200, n_inducing=50, thresh=1e-6,
                 verbose=False, use_loader=False, batches=20, standardize=True):
    torch.manual_seed(seed)
    print(f"Run: {seed}")

    f0 = LogisticGPVI(y, X, n_inducing=n_inducing, n_iter=n_iter, thresh=thresh, verbose=verbose, 
                            use_loader=use_loader, batches=batches, seed=seed)
    f0.fit()

    f1 = LogisticGPVI(y, X, likelihood=PGLikelihood(), n_inducing=n_inducing, n_iter=n_iter, thresh=thresh, 
                            verbose=verbose, use_loader=use_loader, batches=batches, seed=seed)
    f1.fit()

    return torch.tensor([evaluate_method(f0, X_test, y_test), evaluate_method(f1, X_test, y_test)])


def evaluate_method(func, X_test, y_test):
    y_pred = func.predict(X_test)

    auc = BinaryAUROC()
    auc.update(y_test, y_pred)
    auc= auc.compute().item()

    lower, upper = func.credible_intervals(X_test)
    ci_width = (upper - lower).mean().item()

    return func.runtime, auc, ci_width


