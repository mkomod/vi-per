import torch
import numpy as np
import dsdl

from _00_funcs import process_dataset, evaluate_method_application, analyze_dataset


datasets = ["breast-cancer", "diabetes_scale", "phishing", "svmguide1"]
niters = [200, 200, 100, 20]
use_loader = [False, False, True, True]
RUNS = 50


for dataset, niter, loader in zip(datasets, niters, use_loader):
    y, X, y_test, X_test = process_dataset(dataset)

    res = Parallel(n_jobs=1)(delayed(analyze_dataset)(
            i, y, X, y_test, X_test, n_iter=niter, use_loader=loader
        ) for i in range(1, RUNS+1))

    res = torch.stack(res)
    res = torch.transpose(res, 0, 1)
    torch.save(res, f"../results/real_data/{dataset}.pt")
