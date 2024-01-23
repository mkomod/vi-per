import torch

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from _02_method import LogisticVI
from _97_gpytorch import LogisticGPVI, LogitLikelihoodMC, PGLikelihood, LogitLikelihood

from torcheval import metrics as tm

from joblib import Parallel, delayed

# qsub -I -l select=1:ncpus=40:mem=80gb -l walltime=08:00:00 
# export OMP_NUM_THREADS=40


# load in the data
data = pd.read_csv("../data/data_add_std_fillNaNs.csv")
data.head()
data.shape

y = data.liq
X = data[["PGV", "Vs30", "precip", "dw", "wtd"]]
X = data[['lnPGV', 'lnVs30', 'lnprecip', 'lndw', 'lnwtd']]
quake = "lomaprieta1989"

X_train = torch.tensor(X[data.earthquake != quake].values, dtype=torch.double)
y_train = torch.tensor(y[data.earthquake != quake].values, dtype=torch.double)

# centre the dataset
X_m = torch.mean(X_train, dim=0)
X_train = (X_train - X_m) 

X_test =  torch.tensor(X[data.earthquake == quake].values, dtype=torch.double)
X_test = (X_test - X_m)
y_test =  torch.tensor(y[data.earthquake == quake].values, dtype=torch.double)
dat = {"X": X_train, "y": y_train}


f0 = LogisticVI(dat, method=0, n_iter=1500, verbose=True, intercept=True, lr=0.05)
f0.fit()
f0.y, f0.X, f0.XX = None, None, None
torch.save(f0, "../results/application/f0.pt")

f1 = LogisticVI(dat, method=1, n_iter=2000, verbose=True, intercept=True, lr=0.05)
f1.fit()
f1.y, f1.X = None, None
torch.save(f1, "../results/application/f1.pt")

f2 = LogisticVI(dat, method=2, n_iter=1000, verbose=True, intercept=True)
f2.fit()
f2.y, f2.X = None, None
torch.save(f2, "../results/application/f2.pt")

f3 = LogisticVI(dat, method=3, n_iter=1000, verbose=True, intercept=True)
f3.fit()
f3.y, f3.X = None, None
torch.save(f3, "../results/application/f3.pt")

f4 = LogisticVI(dat, method=4, n_iter=300, verbose=True, intercept=True, lr=0.10, n_samples=200) 
f4.fit()
f4.y, f4.X, f4.XX = None, None, None
torch.save(f4, "../results/application/f4.pt")

f5 = LogisticVI(dat, method=5, n_iter=150, verbose=True, intercept=True, lr=0.10, n_samples=200)
f5.fit()
f5.y, f5.X = None, None
torch.save(f5, "../results/application/f5.pt")


del dat, X, y


f0 = torch.load("../results/application/f0.pt")
f1 = torch.load("../results/application/f1.pt")
f2 = torch.load("../results/application/f2.pt")
f3 = torch.load("../results/application/f3.pt")
f4 = torch.load("../results/application/f4.pt")
f5 = torch.load("../results/application/f5.pt")

res = []

for f in [f0, f1, f4, f5]:
    auc = tm.BinaryAUROC()
    auc.update(f.predict(X_test), y_test)
    auc_test = auc.compute().item()
    
    auc = tm.BinaryAUROC()
    auc.update(f.predict(X_train), y_train)
    auc_train = auc.compute().item()

    # add intercept
    f.X = torch.cat([torch.ones(X_train.shape[0], 1), X_train], dim=1)
    f.y = y_train
    f.XX = f.X**2
    
    elbo = f._ELBO_MC(50).item()

    ci_width = torch.median(torch.diff(f.credible_intervals())).item()

    res.append([elbo, ci_width, auc_train, auc_test])


for f in [f2, f3]:
    auc = tm.BinaryAUROC()
    auc.update(f.predict(X_test), y_test)
    auc_test = auc.compute().item()
    
    auc = tm.BinaryAUROC()
    auc.update(f.predict(X_train), y_train)
    auc_train = auc.compute().item()

    f.X = torch.cat([torch.ones(X_train.shape[0], 1), X_train], dim=1)
    f.y = y_train
    f.XX = f.X**2
    
    elbo = f._ELBO_MC(50).item()

    ci_width = torch.median(torch.diff(f.credible_intervals())).item()

    res.append([elbo, ci_width, auc_train, auc_test])


res = torch.tensor(res)
for i in [0, 2, 4, 1, 3, 5]:
    print(f"{res[i, 0]:.2f} & {res[i, 1]:.4f} & {res[i, 2]:.4f} & {res[i, 3]:.4f} \\\\")

