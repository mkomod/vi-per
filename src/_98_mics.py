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

    return torch.tensor([f0._ELBO_MC(), f1._ELBO_MC()])

res = []

for l in range(1, LS+1):
    temp = Parallel(n_jobs=5)(delayed(run_experiment)(i, l) for i in range(1, RUNS+1))
    temp = torch.stack(temp)
    temp = torch.transpose(temp, 0, 1)
    res.append(temp)

r0 = torch.vstack([r[0, :] for r in res])
r1 = torch.vstack([r[1, :] for r in res])

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].boxplot(r0.t().detach().numpy(), showfliers=False, labels=list(range(1, LS+1)))
ax[1].boxplot(r1.t().detach().numpy(), showfliers=False, labels=list(range(1, LS+1)))

plt.show()



# runtime for methods as n increases
DGP=0
NS = [500, 1000, 1500, 2000, 2500, 3500, 5000, 7500, 10000, 12500, 15000, 20000]
RUNS = 10
P = 10

def run_experiment(seed, n):
    print(f"Experiment {seed}")
    dat = generate_data(n, P, seed=seed, dgp=DGP)
    
    f0 = LogisticVI(dat, method=0, intercept=False)
    f1 = LogisticVI(dat, method=1, intercept=False)
    f2 = LogisticVI(dat, method=2, intercept=False)
    f3 = LogisticVI(dat, method=3, intercept=False)
    
    f0.fit()
    f1.fit() 
    f2.fit()
    f3.fit()

    return torch.tensor([evaluate_method(f0, dat), 
                         evaluate_method(f1, dat), 
                         evaluate_method(f2, dat),
                         evaluate_method(f3, dat)])


res = []

for n in NS:
    temp = Parallel(n_jobs=-2)(delayed(run_experiment)(i, n) for i in range(1, RUNS+1))
    temp = torch.stack(temp)
    temp = torch.transpose(temp, 0, 1)
    res.append(temp)

torch.save(res, "../results/runtime.pt")

r0 = torch.vstack([r[0, :, 4] for r in res])
r1 = torch.vstack([r[1, :, 4] for r in res])
r2 = torch.vstack([r[2, :, 4] for r in res])
r3 = torch.vstack([r[3, :, 4] for r in res])

NS
"  ".join([f"{i:.3f}" for i in r0.median(1)[0]])
"  ".join([f"{i:.3f}" for i in r1.median(1)[0]])
"  ".join([f"{i:.3f}" for i in r2.median(1)[0]])
"  ".join([f"{i:.3f}" for i in r3.median(1)[0]])


plt.plot(r0.mean(1).detach().numpy())
plt.plot(r1.mean(1).detach().numpy())
plt.plot(r2.mean(1).detach().numpy())
plt.plot(r3.mean(1).detach().numpy())
plt.legend(["TB-D", "TB-F", "JJ-D", "JJ-F"])
# plt.yscale("log")
plt.show()