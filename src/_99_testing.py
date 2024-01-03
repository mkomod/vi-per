import math

import torch
import torch.autograd as autograd
import torch.distributions as dist

from importlib import reload

from _01_data_generation import generate_data
from _02_method import *
from _00_funcs import *



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


from importlib import reload
import sys
reload(sys.modules["_02_method"])

dat = generate_data(1000, 25, dgp=2, seed=99)

f0 = LogisticVI(dat, method=0, intercept=False, verbose=True)
f0.fit()
f0.runtime

f1 = LogisticVI(dat, method=1, intercept=False, verbose=True, n_iter=1500)
f1.fit()
f1.runtime

f6 = LogisticMCMC(dat, intercept=False, n_iter=1000, burnin=500, verbose=True)
f6.fit()

from torch.profiler import profile, record_function, ProfilerActivity
f1 = LogisticVI(dat, method=1, intercept=False, verbose=True, n_iter=1500)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        f1.fit()


print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))



#
#
# 

def nb_2(m, s, l_max = 10.0):
    l = torch.arange(1.0, l_max*2, 1.0, requires_grad=False, dtype=torch.float64)
    l = l.unsqueeze(0)

    res = torch.sum(
            s / torch.sqrt(2.0 * torch.tensor(torch.pi)) * torch.exp(- 0.5 * m**2 / s**2) + \
            m * ndtr(m / s)
        ) + \
        torch.sum(
            (-1.0)**(l - 1.0) / l * (
                torch.exp( m @ l + 0.5 * s**2 @ (l ** 2) + log_ndtr(-m / s - s @ l)) + \
                torch.exp(-m @ l + 0.5 * s**2 @ (l ** 2) + log_ndtr( m / s - s @ l))
            )
        )
    return res


dat = generate_data(200, 5, dgp=2, seed=99)
f0 = LogisticVI(dat, method=0, intercept=False, verbose=True, l_max=12.0)
f0.fit()

f_pred = dat["X"] @ f0.sample(5000).t()
f_pred.mean(1).shape

f6 = LogisticMCMC(dat, intercept=False, n_iter=1000, burnin=500, verbose=True)
f6.fit()

f_pred = dat["X"] @ f6.B.t()
f_pred.mean(1).shape

evaluate_method(f0, dat)
evaluate_method(f6, dat)


m = torch.tensor(-1.0, dtype=torch.double, requires_grad=True)
s = torch.tensor(2.0, dtype=torch.double, requires_grad=True)

nor = dist.Normal(m, s)
samp = nor.sample((5000, ))

plt.hist(torch.log1p(torch.exp(samp)).detach().numpy(), bins=100)
plt.show()

m = m.unsqueeze(0)
s = s.unsqueeze(0)

for l in range(1, 20):
    nb(m, s, l_max=l)

mc_est(m, s, n_samples=1000000)

res = []
res0 = []
for l in range(1, 20):
    m = torch.tensor(0.152, dtype=torch.double, requires_grad=True)
    s = torch.tensor(0.486, dtype=torch.double, requires_grad=True)
    r = nb(m, s, l_max=l)
    r.backward()
    res.append(s.grad.item())
    
    m = torch.tensor(0.155, dtype=torch.double, requires_grad=True)
    s = torch.tensor(0.498, dtype=torch.double, requires_grad=True)
    r = nb(m, s, l_max=l)
    r.backward()
    res0.append(s.grad.item())
    

plt.plot(range(1, 20), res)
plt.plot(range(1, 20), res0)
plt.show()

res

dat = generate_data(1000, 5, dgp=0, seed=9)
f = LogisticVI(dat, method=0, intercept=False, verbose=True, l_max=12.0)
f.fit()
evaluate_method(f, dat)




MM = dat["X"] @ f.m

plt.hist(MM.detach().numpy(), bins=100)
plt.show()

SS = dat["X"] ** 2 @ f.s**2
plt.hist(SS.detach().numpy(), bins=100)
plt.show()
SS.mean()
MM.mean()

