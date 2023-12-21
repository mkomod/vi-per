import torch
import numpy as np
import dsdl

from joblib import Parallel, delayed

from _00_funcs import process_dataset, evaluate_method_application, analyze_dataset


datasets = ["breast-cancer", "diabetes_scale", "svmguide1", "splice", "australian", "german.numer", "fourclass", "heart"]
niters = [500, 500, 500, 500, 500, 500, 500, 500]
use_loader = [False, False, False, False, False, False, False, False]

RUNS = 100
CPUS = -2

for dataset, niter, loader in zip(datasets, niters, use_loader):
    y, X, y_test, X_test = process_dataset(dataset)

    res = Parallel(n_jobs=CPUS)(delayed(analyze_dataset)(
            i, y, X, y_test, X_test, n_iter=niter, use_loader=loader, lr=0.03, thresh=1e-7
        ) for i in range(1, RUNS+1))

    res = torch.stack(res)
    res = torch.transpose(res, 0, 1)
    torch.save(res, f"../results/real_data/{dataset}.pt")


