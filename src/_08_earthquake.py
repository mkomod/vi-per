import torch

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from _02_method import LogisticVI
from _97_gpytorch import LogisticGPVI, LogitLikelihoodMC, PGLikelihood, LogitLikelihood

from torcheval import metrics as tm

# qsub -I -l select=1:ncpus=40:mem=80gb -l walltime=08:00:00 
# export OMP_NUM_THREADS=40


# load in the data
data = pd.read_csv("../data/data_add_std_fillNaNs.csv")
data.head()
data.shape


y = data.liq
X = data[["PGV", "Vs30", "precip", "dw", "wtd"]]

quake = "lomaprieta1989"

X_train = torch.tensor(X[data.earthquake != quake].values, dtype=torch.double)
y_train = torch.tensor(y[data.earthquake != quake].values, dtype=torch.double)

# sample 30% of the data for training
torch.manual_seed(1)
idx = torch.randperm(X_train.size()[0])
idx = idx[:int(0.3*X_train.size()[0])]
X_train = X_train[idx, :]
y_train = y_train[idx]

X_test =  torch.tensor(X[data.earthquake == quake].values, dtype=torch.double)
y_test =  torch.tensor(y[data.earthquake == quake].values, dtype=torch.double)

dat = {"X": X_train, "y": y_train}

# fit the model
# f0 = LogisticVI(dat, method=0, n_iter=2000, verbose=True, intercept=True, lr=0.03)
# f0.fit()
# torch.save(f0, "../results/application/f0.pt")
# 
# f1 = LogisticVI(dat, method=1, n_iter=2000, verbose=True, intercept=True, lr=0.03)
# f1.fit()
# torch.save(f1, "../results/application/f1.pt")
# 
# f2 = LogisticVI(dat, method=2, n_iter=1000, verbose=True, intercept=True)
# f2.fit()
# torch.save(f2, "../results/application/f2.pt")
# 
# f3 = LogisticVI(dat, method=3, n_iter=1000, verbose=True, intercept=True)
# f3.fit()
# torch.save(f3, "../results/application/f3.pt")

f0 = torch.load("../results/application/f0.pt")
f1 = torch.load("../results/application/f1.pt")
f2 = torch.load("../results/application/f2.pt")
f3 = torch.load("../results/application/f3.pt")

for f in [f0, f1, f2, f3]:
    print(f._ELBO_MC(n_samples=20).item())
    auc = tm.BinaryAUROC()
    auc.update(f.predict(X_test), y_test)
    print(auc.compute().item())


f0.m
f1.m
f2.m
f3.m

