import torch

from _00_funcs import evaluate_method
from _01_data_generation import generate_data
from _02_method  import LogisticVI, LogisticMCMC

from argparse import ArgumentParser
from joblib import Parallel, delayed

parser = ArgumentParser()
parser.add_argument("-d", "--dgp", type=int, default=0)
parser.add_argument("-n", "--n", type=int, default=0)
parser.add_argument("-p", "--p", type=int, default=0)
parser.add_argument("-r", "--runs", type=int, default=100)
args = parser.parse_args()

DGP = args.dgp
N = [250, 500, 1000][args.n]
P = [10, 20, 50][args.p]
RUNS = args.runs
RUNS=4

def run_experiment(seed):
    print(f"Experiment {seed}")
    dat = generate_data(N, P, seed=seed, dgp=DGP)
    
    f0 = LogisticVI(dat, method=0, intercept=False, n_iter=1000)
    f1 = LogisticVI(dat, method=1, intercept=False, n_iter=1000)
    f2 = LogisticVI(dat, method=2, intercept=False, n_iter=1000)
    f3 = LogisticVI(dat, method=3, intercept=False, n_iter=1000)
    f4 = LogisticVI(dat, method=4, intercept=False, n_iter=1000, n_samples=250)
    f5 = LogisticVI(dat, method=5, intercept=False, n_iter=1000, n_samples=250)
    f6 = LogisticMCMC(dat, intercept=False, n_iter=10000, burnin=5000, k=10)

    f0.fit(); f1.fit(); f2.fit(); f3.fit(); f4.fit(); f5.fit(); f6.fit()

    return torch.tensor([evaluate_method(f0, dat),
                         evaluate_method(f1, dat),
                         evaluate_method(f2, dat), 
                         evaluate_method(f3, dat),
                         evaluate_method(f4, dat),
                         evaluate_method(f5, dat),
                         evaluate_method(f6, dat, method="mcmc")])


res = Parallel(n_jobs=-2)(delayed(run_experiment)(i) for i in range(1, RUNS+1))
res = torch.stack(res)
res = torch.transpose(res, 0, 1)


torch.save(res, f"../results/res_{DGP}_{args.n}_{args.p}.pt")