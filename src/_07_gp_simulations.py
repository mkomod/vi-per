import math

import torch
import numpy as np
import matplotlib.pyplot as plt
import dsdl
import gpytorch
import tqdm

from torcheval.metrics import BinaryAUROC
from joblib import Parallel, delayed


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
                            use_loader=use_loader, batches=batches, seed=seed, lr=0.08)
    f0.fit()

    f1 = LogisticGPVI(train_y, train_x, likelihood=LogitLikelihoodMC(), n_inducing=n_inducing, n_iter=n_iter, thresh=thresh,
                            verbose=verbose, use_loader=use_loader, batches=batches, seed=seed, lr=0.05)
    f1.fit()

    f2 = LogisticGPVI(train_y, train_x, likelihood=PGLikelihood(), n_inducing=n_inducing, n_iter=n_iter, thresh=thresh, 
                            verbose=verbose, use_loader=use_loader, batches=batches, seed=seed, lr=0.08)
    f2.fit()

    return torch.tensor([
        evaluate_method_simulation(f0, train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f),
        evaluate_method_simulation(f1, train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f),
        evaluate_method_simulation(f2, train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f),
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
    pred_y = func.predict(test_x)

    auc = BinaryAUROC()
    auc.update(pred_y, test_y)
    auc_test = auc.compute().item()

    pred_y_2 = func.predict(train_x)
    auc.update(pred_y_2, train_y)
    auc_train = auc.compute().item()

    lower, upper = func.credible_intervals(test_x)
    ci_width = (upper - lower).mean().item()

    n = test_y.size()[0]
    coverage_f = torch.sum( (test_f > lower) & (test_f < upper) ) / n

    samp = func.model(test_x).sample(torch.Size([1000]))
    p0 = torch.sigmoid(samp)

    lower_p = torch.quantile(p0, 0.025, dim=0)
    upper_p = torch.quantile(p0, 0.975, dim=0)
    
    coverage_p = torch.sum( (test_p > lower_p) & (test_p < upper_p) ) / n

    f_pred = func.model(xs).mean
    mse = ((true_f.reshape(-1) - f_pred) ** 2).mean().item()
    
    return func.runtime, mse, ci_width, \
            auc_train, auc_test, \
            func.neg_log_likelihood().item(), func.neg_log_likelihood(test_x, test_y).item(), \
            func.log_marginal().item(),             func.log_marginal(test_x, test_y).item(), \
            func.ELB0_MC().item(),                       func.ELB0_MC(test_x, test_y).item(), \
            coverage_f.item(), coverage_p.item()

CPUS = -1
RUNS = 100
n = 50


def run_exp(seed):
    train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f = generate_data(n, seed=seed)
    return analyze_simulation(seed, train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f,
     n_iter=1200, n_inducing=50)


res = Parallel(n_jobs=CPUS)(delayed(run_exp)(i) for i in range(1, RUNS+1))
res = torch.stack(res)
res = torch.transpose(res, 0, 1)
torch.save(res, "../results/gp.pt")


# elbo train, elbo test, auc train, auc test, mse, ci width coverage, runtime
metric_order = [-4, -3, 3, 4, 1, 2, -2, 0]
rm = res.median(dim=1)[0]
rl = res.quantile(0.025, dim=1)
ru = res.quantile(0.975, dim=1)
for j in [0, 1, 2]:
    line = ""
    line_comp = [] 
    for i in metric_order:
        if i != 0:
            # line_comp.append(f"{sf(rm[j, i], 3)} ({sf(sd[j, i],  2)})")
            line_comp.append(f"{sf(rm[j, i], 3)} ({sf(rl[j, i],  2)}, {sf(ru[j, i],  2)})")
        else:
            # line_comp.append(f"{seconds_to_hms(float(rm[j, i]))} ({seconds_to_hms(float(sd[j, i]))})")
            line_comp.append(f"{seconds_to_hms(float(rm[j, i]))} ({seconds_to_hms(float(rl[j, i]))}, {seconds_to_hms(float(ru[j, i]))})")
    line += " & ".join(line_comp) + " \\\\"
    print(line)
print()



def run_exp(seed):
    train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f = generate_data(n, seed=seed)
    return eval_convergence(seed, train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f,
     n_iter=1000, n_inducing=50)


res = Parallel(n_jobs=CPUS)(delayed(run_exp)(i) for i in range(1, RUNS+1))
res = torch.stack(res)
res = torch.transpose(res, 0, 1)
torch.save(res, "../results/gp_convergence.pt")

 