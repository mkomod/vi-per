import math

import torch
import numpy as np
import matplotlib.pyplot as plt
import dsdl
import gpytorch
import tqdm

from torcheval.metrics import BinaryAUROC
from joblib import Parallel, delayed

from _00_funcs import sf, seconds_to_hms
from _97_gpytorch import LogisticGPVI, LogitLikelihoodMC, PGLikelihood, LogitLikelihood


def generate_data(n, seed=1):
    torch.manual_seed(seed)
    func = lambda x: - 4.5 * torch.sin(math.pi / 2 * x)
    train_x = torch.cat((torch.linspace(0, 2.5, int(n/2)), torch.linspace(3.5, 5, int(n/2))))

    train_f = func(train_x) + torch.randn(train_x.shape[0]) * 1.0 # * math.sqrt(0.4)
    train_x = train_x.reshape(-1, 1)
    train_p = torch.sigmoid(train_f)
    train_y = torch.bernoulli(train_p)

    test_x = torch.linspace(0, 5, n) 
    test_f = func(test_x) + torch.randn(test_x.shape[0]) * 1.0  # * math.sqrt(0.4)
    test_p = torch.sigmoid((test_f))
    test_y = torch.bernoulli(test_p)

    xs = torch.linspace(0, 5, 100).reshape(-1, 1)
    true_f = func(xs)

    return train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f


def analyze_simulation(seed, train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f,
        n_iter=200, n_inducing=50, thresh=1e-6, lr=0.05, verbose=False, use_loader=False, batches=20):

    torch.manual_seed(seed)
    print(f"Run: {seed}")
        
    f0 = LogisticGPVI(train_y, train_x, n_inducing=n_inducing, n_iter=n_iter, thresh=thresh, verbose=verbose, 
                            use_loader=use_loader, batches=batches, seed=seed, lr=0.05)
    f0.fit()

    f1 = LogisticGPVI(train_y, train_x, likelihood=LogitLikelihoodMC(10000), n_inducing=n_inducing, n_iter=n_iter, thresh=thresh,
                            verbose=verbose, use_loader=use_loader, batches=batches, seed=seed, lr=0.05)
    f1.fit()

    f2 = LogisticGPVI(train_y, train_x, likelihood=PGLikelihood(), n_inducing=n_inducing, n_iter=n_iter, thresh=thresh, 
                            verbose=verbose, use_loader=use_loader, batches=batches, seed=seed, lr=0.05)
    f2.fit()

    kl_0 = torch.distributions.kl.kl_divergence(f0.model(xs), f1.model(xs)).item()
    kl_2 = torch.distributions.kl.kl_divergence(f2.model(xs), f1.model(xs)).item()

    return torch.tensor([
        evaluate_method_simulation(f0, train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f) + [kl_0],
        evaluate_method_simulation(f1, train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f) + [0.0],
        evaluate_method_simulation(f2, train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f) + [kl_2],
    ])



def eval_convergence(seed, train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f,
        n_iter=200, n_inducing=50, thresh=1e-6, lr=0.05, verbose=False, use_loader=False, batches=20):
        
    torch.manual_seed(seed)
    print(f"Run: {seed}")
        
    f0 = LogisticGPVI(train_y, train_x, n_inducing=n_inducing, n_iter=n_iter, thresh=thresh, verbose=verbose, 
                            use_loader=use_loader, batches=batches, seed=seed, lr=0.08)
    f0.fit()

    f1 = LogisticGPVI(train_y, train_x, likelihood=LogitLikelihoodMC(), n_inducing=n_inducing, n_iter=n_iter, thresh=thresh,
                            verbose=verbose, use_loader=use_loader, batches=batches, seed=seed, lr=0.05)
    f1.fit()

    f2 = LogisticGPVI(train_y, train_x, likelihood=PGLikelihood(), n_inducing=n_inducing, n_iter=n_iter, thresh=thresh, 
                            verbose=verbose, use_loader=use_loader, batches=batches, seed=seed, lr=0.08)
    f2.fit()

    # the losses are not the same length - so we will
    # pad them with None values
    l0 = f0.loss
    l1 = f1.loss
    l2 = f2.loss

    max_len = n_iter
    l0 = l0 + [-100] * (max_len - len(l0))
    l1 = l1 + [-100] * (max_len - len(l1))
    l2 = l2 + [-100] * (max_len - len(l2))

    return torch.tensor([
        l0, l1, l2
    ])



def evaluate_method_simulation(func, train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f):
    auc = BinaryAUROC()
    pred_y_2 = func.predict(train_x)
    auc.update(pred_y_2, train_y)
    auc_train = auc.compute().item()
    
    auc = BinaryAUROC()
    pred_y = func.predict(test_x)
    auc.update(pred_y, test_y)
    auc_test = auc.compute().item()

    true_f = true_f.reshape(-1)
    n = true_f.size()[0]
    f_pred = func.model(xs).mean
    lower, upper = func.credible_intervals(xs)

    mse = ((true_f - f_pred) ** 2).mean().item()
    coverage_f = torch.sum( (true_f > lower) & (true_f < upper) ) / n
    ci_width = (upper - lower).mean().item()

    return [func.ELB0_MC().item(), func.ELB0_MC(test_x, test_y).item(), \
            auc_train, auc_test, \
            mse, ci_width, coverage_f.item(), \
            func.runtime]

CPUS = -1
RUNS = 100
n = 50
torch.manual_seed(1)

def run_exp(seed):
    train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f = generate_data(n, seed=seed)
    return analyze_simulation(seed, train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f,
     n_iter=1500, n_inducing=50)


res = Parallel(n_jobs=CPUS)(delayed(run_exp)(i) for i in range(1, RUNS+1))
res = torch.stack(res)
res = torch.transpose(res, 0, 1)
torch.save(res, "../results/gp.pt")


res = torch.load("../results/gp.pt")
rm = res.median(dim=1)[0]
rl = res.quantile(0.025, dim=1)
ru = res.quantile(0.975, dim=1)

for j in [0, 1, 2]:
    line = ""
    line_comp = [] 
    for i in range(9):
        if i != 7:
            line_comp.append(f"{sf(rm[j, i], 3)} ({sf(rl[j, i],  2)}, {sf(ru[j, i],  2)})")
        else:
            line_comp.append(f"{seconds_to_hms(float(rm[j, i]))} ({seconds_to_hms(float(rl[j, i]))}, {seconds_to_hms(float(ru[j, i]))})")
    line += " & ".join(line_comp) + " \\\\"
    print(line)
print()


# --------------------------------------------------
#        Analyze the convergence
# --------------------------------------------------
# def run_exp(seed):
#     train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f = generate_data(n, seed=seed)
#     return eval_convergence(seed, train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f,
#      n_iter=1500, n_inducing=50)
# 
# res = Parallel(n_jobs=CPUS)(delayed(run_exp)(i) for i in range(1, RUNS+1))
# res = torch.stack(res)
# res = torch.transpose(res, 0, 1)
# torch.save(res, "../results/gp_convergence.pt")
#  