import torch

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from _02_method import LogisticVI
from _97_gpytorch import LogisticGPVI, LogitLikelihoodMC, PGLikelihood, LogitLikelihood


# load in the data
data = pd.read_csv("../data/data_add_std_fillNaNs.csv")
data.head()
data.shape

y = data.liq
X = data[["PGV", "Vs30", "precip", "dw", "wtd"]]

quake = "lomaprieta1989"
X_train = torch.tensor(X[data.earthquake != quake].values, dtype=torch.double)
y_train = torch.tensor(y[data.earthquake != quake].values, dtype=torch.double)
X_test =  torch.tensor(X[data.earthquake == quake].values, dtype=torch.double)
y_test =  torch.tensor(y[data.earthquake == quake].values, dtype=torch.double)

dat = {"X": X_train, "y": y_train}


# fit the model
f0 = LogisticVI(dat, method=0, n_iter=300, verbose=True, intercept=True)

f1 = LogisticVI(dat, method=1, n_iter=1000, verbose=True, intercept=True)
f1.fit()

