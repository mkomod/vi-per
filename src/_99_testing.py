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
# todo fix:
autograd.gradcheck(ELBO_MC, (m, s, dat["y"], dat["X"], mu, sig)) 


m = torch.randn(4, requires_grad=True, dtype=torch.double)
u = torch.ones(4, requires_grad=True, dtype=torch.double)
t = torch.ones(250, dtype=torch.double, requires_grad=True)
mu = torch.zeros(4, dtype=torch.double)
Sig = torch.eye(4, dtype=torch.double)
sig = torch.ones(4, dtype=torch.double) * 4

ELBO_Jak_mvn(m, u, t, dat["y"], dat["X"], mu, Sig)


m
torch.exp(u)
t

dat["b"]
m

S = torch.inverse(torch.cov(dat["X"].t()) + torch.diag(torch.exp(u)))

f = LogisticVI(dat, intercept=False, method=0, seed=0, l_max=10.0)
f.fit()
f.s
f.m
f.ELBO()



# jaakkola analytical solutions


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

S_0 = torch.diag(sig)
a_t = (torch.sigmoid(t) - 0.5) / t
S_N = torch.inverse(torch.inverse(S_0) + X.t() @ torch.diag(a_t) @ X)
    
for i in range(300):
    a_t = (torch.sigmoid(t) - 0.5) / t
    # full covariance matrix case
    m = S_N @ (torch.inverse(S_0) @ mu + X.t() @ (y - 0.5))
    S_N = torch.inverse(torch.inverse(S_0) + X.t() @ torch.diag(a_t) @ X)
    t = torch.sqrt(torch.diag(X @ (S_N  + torch.outer(m, m)) @ X.t()))


for i in range(300):
    a_t = (torch.sigmoid(t) - 0.5) / t
    # full covariance matrix case
    m = torch.inverse(X.t() @ torch.diag(a_t) @ X + 1/sig**2) @ (mu / sig**2 + X.t() @ (y - 0.5))
    s = torch.sqrt(1 / (1/sig**2 + torch.diag(X.t() @ torch.diag(a_t) @ X)))
    t = torch.sqrt(torch.diag(X @ (torch.diag(s**2)  + torch.outer(m, m)) @ X.t()))


f = LogisticVI(dat, intercept=False, method=3, seed=0, l_max=10.0)
f.fit() 
f.m

m = torch.randn(4, requires_grad=True, dtype=torch.double)
u = torch.ones(4, requires_grad=True, dtype=torch.double)
t = torch.ones(250, dtype=torch.double, requires_grad=True)
s = torch.exp(u)





def ELL_Jak(m, s, t, y, X):
    """
    Compute the expected negative log-likelihood using the bound introduced
    by Jaakkola and Jordan (2000)
    :return: ELL
    """
    M = X @ m
    a_t = (torch.sigmoid(t) - 0.5) / t
    B = a_t * (X **2 @ (s ** 2 + m ** 2))

    res = - torch.dot(y, M) - torch.sum(logsigmoid(t)) + \
        0.5 * torch.sum(M + t) + 0.5 * torch.sum(B)   - \
        0.5 * torch.sum(a_t * t ** 2)

    return res

def KL(m, s, mu, sig):
    """
    Compute the KL divergence between two Gaussians
    :param m: mean of variational distribution
    :param s: standard deviation of variational distribution
    :param mu: mean of prior
    :parma sig: standard deviation of prior
    :return: KL divergence
    """
    res = torch.log(sig / s) + 0.5 * ((s ** 2 + (m - mu) ** 2) / sig ** 2 - 1)
    return torch.sum(res)

def ELBO_Jak(m, u, t, y, X, mu, sig):
    """
    Compute the negative of the ELBO using the bound introduced by
    Jaakkola and Jordan (2000)
    :return: ELBO
    """
    s = torch.exp(u)
    return ELL_Jak(m, s, t, y, X) + KL(m, s, mu, sig)



torch.manual_seed(1)
dat = generate_data(250, 4)


# test other formulation of cov matrix
dat = generate_data(250, 4)

m = torch.randn(4, requires_grad=True, dtype=torch.double)
u = torch.ones(4, requires_grad=True, dtype=torch.double)
t = torch.ones(250, dtype=torch.double, requires_grad=True)

mu = torch.zeros(4, dtype=torch.double)
Sig = torch.eye(4, dtype=torch.double)
sig = torch.ones(4, dtype=torch.double) * 4




optimizer = torch.optim.SGD([m, u], lr=0.01, momentum=0.9)
loss = []

# training loop
for epoch in range(500):
    optimizer.zero_grad()
    # l = ELBO_TB(m, u, dat["y"], dat["X"], mu, sig)
    l = ELBO_TB_mvn(m, u, t, dat["y"], dat["X"], mu, Sig, l_max=10.0)
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
    l = ELBO_TB_mvn(m, u, dat["y"], dat["X"], mu, Sig, l_max=20.0)
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