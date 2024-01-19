import torch
from _00_funcs import print_results, sf, seconds_to_hms

N = ["500", "1000", "10000", "20000"]
P = ["5", "10", "25"]
METHOD = ["TB-D", "TB-F", "JJ-D", "JJ-F", "MC-D", "MC-F", "MCMC"]
metric_order = range(9)
metric_order = [0, 1, 5, 6, 7, 8]

for p in range(2, 3):
    for n in range(1, 3):
        for dgp in range(2, 3):
            print("\\multirow{7}{*}{" + f"{N[n]} / {P[p]}" + "}")
            res = torch.load(f"../results/res_{dgp}_{n}_{p}.pt")
            rm = res.median(dim=1)[0]
            rl = res.quantile(0.025, dim=1)
            ru = res.quantile(0.975, dim=1)
            for j in [0, 4, 2, 1, 5, 3, 6]:
                line = ""
                line_comp = [] 
                for i in metric_order:
                    if i != 8:
                        line_comp.append(f"{sf(rm[j, i], 3)} ({sf(rl[j, i],  2)}, {sf(ru[j, i],  2)})")
                    else:
                        line_comp.append(f"{seconds_to_hms(float(rm[j, i]))} ({seconds_to_hms(float(rl[j, i]))}, {seconds_to_hms(float(ru[j, i]))})")
                line += " & ".join(line_comp) + " \\\\"
                print(line)
            print()



# datasets = ["breast-cancer", "diabetes_scale", "svmguide1", "splice", "australian"] # , "fourclass", "heart"]
# datasets = ["breast-cancer", "diabetes_scale", "svmguide1", "splice", "australian", "fourclass", "heart"]
datasets = ["breast-cancer", "svmguide1", "australian", "fourclass", "heart"]

# elbo train, elbo test, auc trian, auc test, coverage, time
# metric_order = [-4, 2, 3, 1, -1, -2, 0]
metric_order = [-4, -2, -1, 1, 2, 3, 0]

for dataset in datasets:
    res = torch.load(f"../results/real_data/{dataset}.pt")
    print("\n" + dataset)
    rm = res.median(dim=1)[0]
    rl = res.quantile(0.025, dim=1)
    ru = res.quantile(0.975, dim=1)
    for j in [0, 1, 2]:
        line = ""
        line_comp = [] 
        for i in metric_order:
            if i == 0:
                line_comp.append(f"{seconds_to_hms(float(rm[j, i]))} ({seconds_to_hms(float(rl[j, i]))}, {seconds_to_hms(float(ru[j, i]))})")
            elif i != -4:
                line_comp.append(f"{sf(rm[j, i], 3)} ({sf(rl[j, i],  3)}, {sf(ru[j, i],  3)})")
            else:
                line_comp.append(f"{sf(rm[j, i], 4)} ({sf(rl[j, i],  4)}, {sf(ru[j, i],  4)})")
        line += " & ".join(line_comp) + " \\\\"
        print(line)
    print()



res = torch.load("../results/gp.pt")
metric_order = range(9)
rm = res.median(dim=1)[0]
rl = res.quantile(0.10, dim=1)
ru = res.quantile(0.90, dim=1)
for j in [0, 1, 2]:
    line = ""
    line_comp = [] 
    for i in metric_order:
        if i == 7:
            line_comp.append(f"{seconds_to_hms(float(rm[j, i]))} ({seconds_to_hms(float(rl[j, i]))}, {seconds_to_hms(float(ru[j, i]))})")
        else:
            line_comp.append(f"{sf(rm[j, i], 3)} ({sf(rl[j, i],  2)}, {sf(ru[j, i],  2)})")
    line += " & ".join(line_comp) + " \\\\"
    print(line)
print()


res = torch.load("../results/gp_convergence.pt")

(1500.0 - (res == -100).sum(dim=2)).mean(dim=1)