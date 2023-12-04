import os
import torch
from _97_gpytorch import LogisticGPVI
import numpy as np
import dsdl
import urllib
import matplotlib.pyplot as plt


ds = dsdl.load("cod-rna")
X, y = ds.get_train()

X = X.todense()
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

y[y == -1] = 0

f = LogisticGPVI(y, X, verbose=True, n_iter=350)
f.fit()

plt.plot(f.loss)
plt.show()

X_test, y_test = ds.get_test()
X_test = X_test.todense()
y_test[y_test == -1] = 0

id1 = np.random.choice(np.where(y_test == 1)[0], 1000, replace=False)
id0 = np.random.choice(np.where(y_test == 0)[0], 1000, replace=False)

y_test = np.concatenate((y_test[id1], y_test[id0]))
X_test = X_test[np.concatenate((id1, id0)), :]

y_test = torch.tensor(y_test, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)

y_pred = f.predict(X_test)


import torcheval as tm

tm.binary_classification_metrics(y_test, y_pred)

torch.sum(y_pred.ge(0.5) == y_test)

