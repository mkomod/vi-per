import torch
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from _00_funcs import evaluate_method
from _01_data_generation import generate_data
from _02_method import LogisticVI

N = 1000
P = 5
DGP = 0
RUNS = 100
LS = 12

def run_experiment(seed, l):
    print(f"Experiment {seed}")
    dat = generate_data(N, P, seed=seed, dgp=DGP)
    
    f0 = LogisticVI(dat, method=0, intercept=False, l_max=l)
    f1 = LogisticVI(dat, method=1, intercept=False, l_max=l) 

    f0.fit()
    f1.fit() 

    return torch.tensor([evaluate_method(f0, dat), evaluate_method(f1, dat)])

res = []

for l in range(1, LS+1):
    temp = Parallel(n_jobs=5)(delayed(run_experiment)(i, l) for i in range(1, RUNS+1))
    temp = torch.stack(temp)
    temp = torch.transpose(temp, 0, 1)
    res.append(temp)

r0 = torch.vstack([r[0, :, 0] for r in res])
r1 = torch.vstack([r[1, :, 0] for r in res])

plt.boxplot(r0.t().detach().numpy())
plt.boxplot(r1.t().detach().numpy())
plt.yscale("log")
plt.show()


means = torch.stack([torch.mean(i, dim=1) for i in res])




means = torch.stack([torch.mean(i, dim=1) for i in res])
means

fig, axs = plt.subplots(1, 5, figsize=(15, 10))
x_axis = range(1, LS)

for i in range(5):
    axs[i].plot(x_axis, means[:, 0, i].detach().numpy())
    axs[i].plot(x_axis, means[:, 1, i].detach().numpy())

axs[0].legend(["TB-D", "TB-F"])
plt.show()



