import torch
from _00_funcs import print_results

N = ["500", "1000", "10000", "20000"]
P = ["5", "10", "25"]
METHOD = ["TB-D", "TB-F", "JJ-D", "JJ-F", "MC-D", "MC-F", "MCMC"]

for p in range(0, 3):
    print(f"P: {P[p]}")
    for n in range(0, 4):
        for dgp in range(0, 3):
            print(f"DGP: {dgp} N: {N[n]}")
            res = torch.load(f"../results/res_{dgp}_{n}_{p}.pt")
            rm = res.mean(dim=1)

            # rm = res.median(dim=1)[0]
            rl = res.quantile(0.025, dim=1)
            ru = res.quantile(0.975, dim=1)
            for j in [0, 4, 2, 1, 5, 3, 6]:
                if (j == 0):
                    print("DIAG")
                if (j == 1):
                    print("FULL")
                if (j == 6):
                    print("MCMC")
                line = METHOD[j] + " | " +  " | ".join([ f"{rm[j, i]:.3f} ({rl[j, i]:.3f}, {ru[j, i]:.3f})" for i in range(0, 5) ]) # + " \\\\"
                print(line)
            print()


# print results for GP simulation


# print the results for the real datasets
datasets = ["breast-cancer", "diabetes_scale", "phishing", "svmguide1"]

for dataset in datasets:
    res = torch.load(f"../results/real_data/{dataset}.pt")
    print("\n" + dataset)
    # format nicely (auc, runtime, ci)

    print_results(res)
