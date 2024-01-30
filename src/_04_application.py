import torch
import numpy as np
import dsdl

from joblib import Parallel, delayed

from _00_funcs import process_dataset, evaluate_method_application, analyze_dataset, sf, seconds_to_hms


datasets = ["breast-cancer", "svmguide1", "australian", "fourclass", "heart"]
niters = [1500] * len(datasets)
use_loader = [False] * len(datasets)

RUNS = 100
CPUS = -2

for dataset, niter, loader in zip(datasets, niters, use_loader):
    y, X, y_test, X_test = process_dataset(dataset)

    res = Parallel(n_jobs=CPUS)(delayed(analyze_dataset)(
            i, y, X, y_test, X_test, n_iter=niter, use_loader=loader, thresh=1e-6
        ) for i in range(1, RUNS+1))

    res = torch.stack(res)
    res = torch.transpose(res, 0, 1)
    torch.save(res, f"../results/real_data/{dataset}.pt")

