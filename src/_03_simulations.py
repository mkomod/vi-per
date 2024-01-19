import torch

from _00_funcs import evaluate_method
from _01_data_generation import generate_data
from _02_method  import LogisticVI, LogisticMCMC, KL, KL_mvn

from argparse import ArgumentParser
from joblib import Parallel, delayed

parser = ArgumentParser()
parser.add_argument("-d", "--dgp", type=int, default=0)
parser.add_argument("-n", "--n", type=int, default=0)
parser.add_argument("-p", "--p", type=int, default=0)
parser.add_argument("-r", "--runs", type=int, default=100)
args = parser.parse_args()

DGP = args.dgp
N = [500, 1000, 10000, 20000][args.n]
P = [5, 10, 25][args.p]
RUNS = args.runs

print(f"DGP: {DGP}, N: {N}, P: {P}")

def run_experiment(seed):
    print(f"Experiment {seed}")
    dat = generate_data(N, P, seed=seed, dgp=DGP)
    
    f0 = LogisticVI(dat, method=0, intercept=False)
    f1 = LogisticVI(dat, method=1, intercept=False) 
    f2 = LogisticVI(dat, method=2, intercept=False) 
    f3 = LogisticVI(dat, method=3, intercept=False) 
    f4 = LogisticVI(dat, method=4, intercept=False) 
    f5 = LogisticVI(dat, method=5, intercept=False) 
    f6 = LogisticMCMC(dat, intercept=False, n_iter=30000, burnin=25000)

    f0.fit(); f1.fit(); f2.fit(); f3.fit(); f4.fit(); f5.fit(); f6.fit()

    kl_0 = KL(f4.m, f4.s, f0.m, f0.s)
    kl_2 = KL(f4.m, f4.s, f2.m, f2.s)
    kl_1 = KL_mvn(f5.m, f5.S, f1.m, f1.S)
    kl_3 = KL_mvn(f5.m, f5.S, f3.m, f3.S)

    return torch.tensor([evaluate_method(f0, dat) + [kl_0],
                         evaluate_method(f1, dat) + [kl_1],
                         evaluate_method(f2, dat) + [kl_2], 
                         evaluate_method(f3, dat) + [kl_3],
                         evaluate_method(f4, dat) + [-1.0],
                         evaluate_method(f5, dat) + [-1.0],
                         evaluate_method(f6, dat, method="mcmc") + [-1.0]])


res = Parallel(n_jobs=-1)(delayed(run_experiment)(i) for i in range(1, RUNS+1))
res = torch.stack(res)
res = torch.transpose(res, 0, 1)


torch.save(res, f"../results/res_{DGP}_{args.n}_{args.p}.pt")