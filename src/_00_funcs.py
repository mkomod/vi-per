import torch

from torcheval import metrics as tm
import torch.distributions as dist

def evaluate_method(fit, dat, method="vi"):
    auc = tm.BinaryAUROC()
    auc.update(fit.predict(dat["X"]), dat["y"])

    mse = tm.MeanSquaredError()
    creds = fit.credible_intervals()

    if fit.intercept:
        mse.update(fit.m[1:], dat["b"])
        cov = torch.sum(torch.logical_and(creds[1:,0] < dat["b"], dat["b"] < creds[1:,1])) / dat["b"].size()[0]
    else:
        mse.update(fit.m, dat["b"])
        cov = torch.sum(torch.logical_and(creds[:, 0] < dat["b"], dat["b"] < creds[:, 1])) / dat["b"].size()[0]

    cred_size = torch.mean(torch.diff(creds))

    return mse.compute().item(), auc.compute().item(), cov, cred_size, fit.runtime
