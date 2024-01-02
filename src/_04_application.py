import torch
import numpy as np
import dsdl

from joblib import Parallel, delayed

from _00_funcs import process_dataset, evaluate_method_application, analyze_dataset, sf, seconds_to_hms


datasets = ["breast-cancer", "diabetes_scale", "svmguide1", "splice", "australian", "german.numer", "fourclass", "heart"]
niters = [500] * len(datasets)
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


datasets = ["breast-cancer", "svmguide1", "splice", "fourclass", "heart"]
metric_order = [-2, -1, 2, 3, 1, 0]


for dataset in datasets:
    res = torch.load(f"../results/real_data/{dataset}.pt")
    print("\n" + dataset)
    # rm = res.mean(dim=1)
    rm = res.median(dim=1)[0]
    # sd = res.std(dim=1)
    rl = res.quantile(0.025, dim=1)
    ru = res.quantile(0.975, dim=1)
    for j in [0, 1, 2]:
        line = ""
        line_comp = [] 
        for i in metric_order:
            if i != 0:
                line_comp.append(f"{sf(rm[j, i], 4)} ({sf(rl[j, i],  4)}, {sf(ru[j, i],  4)})")
            else:
                line_comp.append(f"{seconds_to_hms(float(rm[j, i]))} ({seconds_to_hms(float(rl[j, i]))}, {seconds_to_hms(float(ru[j, i]))})")
        line += " & ".join(line_comp) + " \\\\"
        print(line)
    print()
