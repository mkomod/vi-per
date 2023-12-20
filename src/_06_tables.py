import torch
from _00_funcs import print_results


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


N = ["500", "1000", "10000", "20000"]
P = ["5", "10", "25"]
METHOD = ["TB-D", "TB-F", "JJ-D", "JJ-F", "MC-D", "MC-F", "MCMC"]
metric_order = [0, 1, 5, 2, 3, 4]


for p in range(1, 2):
    for n in range(1, 3):
        for dgp in range(2, 3):
            # print(f"DGP: {dgp} N: {N[n]} / {P[p]}")
            print("\\multirow{7}{*}{" + f"{N[n]} / {P[p]}" + "}")
            res = torch.load(f"../results/res_{dgp}_{n}_{p}.pt")
            # rm = res.mean(dim=1)
            rm = res.median(dim=1)[0]
            # sd = res.std(dim=1)
            rl = res.quantile(0.025, dim=1)
            ru = res.quantile(0.975, dim=1)
            for j in [0, 4, 2, 1, 5, 3, 6]:
                # line = METHOD[j] + " & " +  " & ".join([f"{sf(rm[j, i], 3)} ({sf(rl[j, i],  2)}, {sf(ru[j, i], 2)})"for i in metric_order ]) + " \\\\"
                # line = " & " + METHOD[j] + " & " 
                line = ""
                line_comp = [] 
                for i in metric_order:
                    if i != 4:
                        # line_comp.append(f"{sf(rm[j, i], 3)} ({sf(sd[j, i],  2)})")
                        line_comp.append(f"{sf(rm[j, i], 3)} ({sf(rl[j, i],  2)}, {sf(ru[j, i],  2)})")
                    else:
                        # line_comp.append(f"{seconds_to_hms(float(rm[j, i]))} ({seconds_to_hms(float(sd[j, i]))})")
                        line_comp.append(f"{seconds_to_hms(float(rm[j, i]))} ({seconds_to_hms(float(rl[j, i]))}, {seconds_to_hms(float(ru[j, i]))})")
                line += " & ".join(line_comp) + " \\\\"
                print(line)
            print()



# print the results for the real datasets
datasets = ["breast-cancer", "diabetes_scale", "svmguide1", "splice", "australian", "german.numer", "fourclass", "heart"]
metric_order = [-2, -1, 1, 2, 0]

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
