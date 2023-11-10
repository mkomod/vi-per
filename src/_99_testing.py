import math

import torch
import torch.autograd as autograd
import torch.distributions as dist

from importlib import reload

from _01_data_generation import generate_data
from _02_method import *



torch.manual_seed(1)

# generate data
dat = generate_data(250, 4)

dat["X"].dtype
dat["y"].dtype
dat["b"].dtype

# set prior params
mu = torch.zeros(4, dtype=torch.double)
sig = torch.ones(4, dtype=torch.double) * 4

# parameters of the variational distribution
m = torch.randn(4, requires_grad=True, dtype=torch.double)
u = torch.tensor([-1.,-1., -1., -1.], requires_grad=True, dtype=torch.double)
s = torch.exp(u)

t = torch.ones(250, dtype=torch.double, requires_grad=True)

# check if values of functions match
KL(m, s, mu, sig)
KL_MC(m, s, mu, sig)
d1 = dist.Normal(mu, sig)
d2 = dist.Normal(m, s)
torch.sum(torch.distributions.kl.kl_divergence(d2, d1))

# check for MultiVariateNormal
KL_mvn(m, torch.diag(s), mu, torch.diag(sig)) 
d1 = dist.MultivariateNormal(mu, torch.diag(sig))  
d2 = dist.MultivariateNormal(m, torch.diag(s))
torch.distributions.kl.kl_divergence(d2, d1)   


ELL_MC(m, s, dat["y"], dat["X"])
ELL_TB(m, s, dat["y"], dat["X"])
ELL_Jak(m, s, t, dat["y"], dat["X"])

ELBO_MC(m, s, dat["y"], dat["X"], mu, sig)
ELBO_TB(m, s, dat["y"], dat["X"], mu, sig)
ELBO_Jak(m, u, t, dat["y"], dat["X"], mu, sig)


autograd.gradcheck(ELBO_TB, (m, s, dat["y"], dat["X"], mu, sig)) 
autograd.gradcheck(ELBO_Jak, (m, s,t,  dat["y"], dat["X"], mu, sig)) 
autograd.gradcheck(ELBO_MC, (m, u, dat["y"], dat["X"], mu, sig)) 


X = dat["X"]
y = dat["y"]


def KL(m, s, mu, sig):
    res = torch.log(sig / s) + 0.5 * ((s ** 2 + (m - mu) ** 2) / sig ** 2 - 1)
    return torch.sum(res)


def ELL_MC(m, s, y, X):
    """
    Compute the expected negative log-likelihood with monte carlo
    :return: ELL
    """
    M = X @ m
    S = torch.sqrt(X ** 2 @ s ** 2)

    with torch.no_grad():    
        norm = dist.Normal(torch.zeros_like(M), torch.ones_like(S))
        samp = norm.sample((1000, ))
        
    samp = M + S * samp

    res =  torch.dot(1 - y, M) + \
        torch.sum(torch.mean(torch.log1p(torch.exp(-samp)), 0))

    return res

def ELBO_MC(m, u, y, X, mu, sig):
    s = torch.exp(u)
    return ELL_MC(m, s, y, X) + KL(m, s, mu, sig)



m = torch.randn(4, requires_grad=True, dtype=torch.double)
u = torch.ones(4, requires_grad=True, dtype=torch.double)
s = torch.exp(u)
mu = torch.zeros(4, dtype=torch.double)
sig = torch.ones(4, dtype=torch.double) * 4

optimizer = torch.optim.SGD([m, u], lr=0.01)
loss = []

# training loop
for epoch in range(1000):
    optimizer.zero_grad()
    l = ELBO_MC(m, u, dat["y"], dat["X"], mu, sig)
    loss.append(l.item())
    l.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(epoch, l.item())

m
torch.exp(u)
dat["b"]
torch.mean((m - dat["b"])**2)

from _02_method import LogisticVI, LogisticMCMC


torch.mean((f0.m- dat["b"])**2)



torch.manual_seed(1)
dat = generate_data(250, 4)

m = torch.randn(4, requires_grad=True, dtype=torch.double)
u = torch.ones(4, requires_grad=True, dtype=torch.double)
t = torch.ones(250, dtype=torch.double, requires_grad=True)
s = torch.exp(u)
mu = torch.zeros(4, dtype=torch.double)
sig = torch.ones(4, dtype=torch.double) * 4
X = dat["X"]
y = dat["y"]

optimizer = torch.optim.SGD([m, u], lr=0.01, momentum=0.9)
loss = []

# training loop
for epoch in range(500):
    optimizer.zero_grad()
    l = ELBO_MC(m, u, dat["y"], dat["X"], mu, sig)
    # l = ELBO_TB_mvn(m, u, t, dat["y"], dat["X"], mu, Sig, l_max=10.0)
    # l = ELBO_Jak(m, u, t, dat["y"], dat["X"], mu, sig)
    # l = ELBO_Jak_mvn(m, u, t, dat["y"], dat["X"], mu, Sig)
    loss.append(l.item())
    l.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(epoch, l.item())








# test other formulation of cov matrix
p = 10
dat = generate_data(250, p)

f = LogisticVI(dat, intercept=False, method=2, seed=0, l_max=10.0)
f.fit()
f.ELBO()


m = torch.randn(p, requires_grad=True, dtype=torch.double)
u = torch.ones(int(p * (p+1.0) / 2.0), dtype=torch.double)
u = u * 1/p
# u = u
u.requires_grad = True
mu = torch.zeros(p, dtype=torch.double)
Sig = torch.eye(p, dtype=torch.double)


# optimizer = torch.optim.SGD([m, u], lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam([m, u], lr=0.01) 
loss = []


# training loop
for epoch in range(500):
    optimizer.zero_grad()
    l = ELBO_MC(m, u, dat["y"], dat["X"], mu, Sig, l_max=20.0)
    loss.append(l.item())
    l.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(epoch, l.item())




def ELBO_TB_mvn(m, u, y, X, mu, Sig, l_max = 10.0, cov = None):
    """
    Compute the negative of the ELBO
    :return: ELBO
    """
    p = Sig.size()[0]
    L = torch.zeros(p, p, dtype=torch.double)
    L[torch.tril_indices(p, p, 0).tolist()] = u
    # L[torch.triu_indices(p, p, 1).tolist()] = u
    # S = L.t() @ L
    S = L @ L.t() 
    return ELL_TB_mvn(m, S, y, X, l_max=l_max) + KL_mvn(m, S, mu, Sig)


p = Sig.size()[0]
L = torch.zeros(p, p, dtype=torch.double)
L[torch.tril_indices(p, p, 0).tolist()] = u
S = L @ L.t()



def f(m, s):
    samp = torch.randn(500, )
    x = samp * s + m
    return torch.mean(torch.log1p(torch.exp(-x)))


m = torch.ones(1, requires_grad=True, dtype=torch.double)
s = torch.ones(1, requires_grad=True, dtype=torch.double) 

o = f(m, s)
o.backward()

m.grad
s.grad

torch.autograd.gradcheck(f, (m, s))

