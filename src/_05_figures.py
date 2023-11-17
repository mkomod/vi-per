import math

import torch
import torch.optim as optim
import torch.distributions as dist

import matplotlib.pyplot as plt
from torch.special import log_ndtr, ndtr
from torch.nn.functional import logsigmoid

import numpy as np

from _02_method import LogisticVI


def a(t):
    return (torch.sigmoid(t) - 0.5) / t


def jb(m, s):
    t = torch.ones_like(m, requires_grad=True)
    opt = optim.LBFGS([t], lr=0.3, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -torch.log(torch.sigmoid(t)) + 0.5 * (m + t) + \
            0.5 * a(t) * (s**2 + m**2 - t**2)
        loss.backward()
        return loss
    
    opt.step(closure)

    return -torch.log(torch.sigmoid(t)) + 0.5 * (m + t) + \
        0.5 * a(t) * (s**2 + m**2 - t**2)


def nb(m, s, l_max = 10.0):
    l = torch.arange(1.0, l_max*2, 1.0, requires_grad=False, dtype=torch.float64)
    l = l.unsqueeze(0)

    res = s / torch.sqrt(2 * torch.tensor(torch.pi)) * torch.exp(- 0.5 * m**2 / s**2) + \
            m * ndtr(m / s) + \
        torch.sum(
            (-1.0)**(l - 1.0) / l * (
                torch.exp( m * l + 0.5 * s**2 * (l ** 2) + log_ndtr(-m / s - s * l)) + \
                torch.exp(-m * l + 0.5 * s**2 * (l ** 2) + log_ndtr( m / s - s * l))
            )
        )
    return res


def mc_est(m, s, n_samples=100000):
    norm = dist.Normal(m, s)
    xs = norm.sample((n_samples,))
    res = torch.mean(torch.log1p(torch.exp(xs)))

    return res



# ---------------------------
#        Figure 1
# ---------------------------
torch.manual_seed(1)
ms = torch.linspace(-3, 3, 250)
s = torch.tensor(2.0)
res0 = torch.zeros((3, 250))

for i, m in enumerate(ms):
    res0[0][i] = jb(m, s)
    res0[1][i] = nb(m, s, 10.0)
    res0[2][i] = mc_est(m, s)


# examine change in the sd
m = torch.tensor(1.0)
ss = torch.linspace(0.1, 3.0, 250)
res1 = torch.zeros(3, 250)

for i, s in enumerate(ss):
    res1[0][i] = jb(m, s)
    res1[1][i] = nb(m, s, 10.0)
    res1[2][i] = mc_est(m, s, n_samples=1000000)


m = torch.linspace(-3, 3, 25)
s = torch.linspace(0.1, 3.0, 30)
m, s = torch.meshgrid(m, s)
m = m.reshape(-1)
s = s.reshape(-1)
grid = torch.stack([m, s], dim=1)
ls = []

# compute the number of terms needed to get the error below 0.01
for m, s in grid:
    l = 1
    res_true = mc_est(m, s, n_samples=2000000)
    while True:
        res = nb(m, s, l_max=l)
        if torch.abs((res_true - res) / res) < 0.01:
            ls.append(l)
            print(f"m={m}, s={s}, l={l}")
            break
        l += 1

# prepare results for plotting
ms = torch.linspace(-3, 3, 250)
ms = ms.detach().numpy()
ss = torch.linspace(0.1, 3.0, 250)
ss = ss.detatch().numpy()

res0 = res0.detach().numpy()
res1 = res1.detach().numpy()

ls = torch.tensor(ls)
ls = ls.reshape(25, 30)
ls = ls.rot90(1)
ls = ls.detach().numpy()


# plot the results
fig, ax = plt.subplots(1, 3, figsize=(15*1.2, 3.5*1.2))
ax[0].plot(ms, res0[0], color="tab:green")
ax[0].plot(ms, res0[1], color="tab:blue")
ax[0].plot(ms, res0[2], color="tab:orange", linestyle="--")
ax[0].grid(alpha=0.2)
ax[0].set_title("(a)  $\\tau = 2.0$", loc="left")
ax[0].set_xlabel("$\\vartheta$")
ax[0].legend(("Jaakkola and Jordan (1999)", "Proposed bound", "Monte-Carlo Estimate"))

ax[1].plot(ss, res1[0], color="tab:green")
ax[1].plot(ss, res1[1], color="tab:blue")
ax[1].plot(ss, res1[2], color="tab:orange", linestyle="--")
ax[1].grid(alpha=0.2)
ax[1].set_title("(b)  $\\vartheta = 1.0$", loc="left")
ax[1].set_xlabel("$\\tau$")

ax[2].matshow(ls, cmap='viridis', interpolation='none', aspect="auto")
for (i, j), z in np.ndenumerate(ls):
    ax[2].text(j, i, '{:d}'.format(z), ha='center', va='center', color="black", fontsize=6)
ax[2].set_xticks([])
ax[2].set_yticks([])
plt.gca().xaxis.tick_bottom()
ax[2].set_xticks(np.arange(0, 25, 4))
ax[2].set_yticks(np.arange(0, 30, 5))
ax[2].set_yticklabels(np.arange(3.0, 0.0, -0.5))
ax[2].set_xticklabels(np.arange(-3, 4, 1))
ax[2].set_xlabel("$\\vartheta$")
ax[2].set_ylabel("$\\tau$")
ax[2].set_title("(c) Value of $l$ such that the relative error is below 1%", loc="left")

# plt.show()
plt.savefig("/home/michael/proj/papers/logistic_vb/figures/error.pdf")


# ---------------------------
#        Figure 2
# ---------------------------
mvn1 = dist.MultivariateNormal(torch.ones(2), torch.eye(2) / 10)
mvn2 = dist.MultivariateNormal(-torch.ones(2), torch.eye(2)/ 10)

X1 = mvn1.sample((50,))
X2 = mvn2.sample((50,))
y1 = torch.ones(50)
y2 = torch.zeros(50)
X = torch.cat([X1, X2], dim=0)
y = torch.cat([y1, y2], dim=0)

X = X.type(torch.double)
y = y.type(torch.double)


f = LogisticVI({"X": X, "y": y}, method=1, intercept=False, l_max=30.0)
f.fit()
f.m
f.ELBO()

# create a grid over [-2, 2] x [-2, 2], which is a n by 2 matrix
XX = torch.linspace(-2, 2, 100)
YY = torch.linspace(-2, 2, 100)
XX, YY = torch.meshgrid(XX, YY)
grid = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)

grid = grid.type(torch.double) 

vals = f.predict(grid)

# plot the contour
plt.contourf(XX.detach().numpy(), YY.detach().numpy(), vals.reshape(100, 100).detach().numpy(), levels=30)
plt.colorbar()
plt.plot(X.detach().numpy()[:50, 0], X.detach().numpy()[:50, 1], "o", color="tab:blue")
plt.plot(X.detach().numpy()[50:, 0], X.detach().numpy()[50:, 1], "o", color="tab:orange")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


# ---------------------------
#        Figure 3
# ---------------------------

# import the data from ../results
# create a  for each of the DGPs and each of the methods

# DGP 0
# NB
res = torch.load("../results/res_0_3_0.pt")
vals = torch.hstack(list(res))
vals = vals.detach().numpy()

vals = []

for metric in range(0, 5):
    temp_vals = []
    for dgp in range(0, 3):
        res = torch.load(f"../results/res_{dgp}_3_2.pt")
        res = torch.hstack(list(res))
        temp_vals.append(res[:, metric::5].detach().numpy())
    vals.append(temp_vals)

# vals is a list of lists, we want to concat these
import numpy as np
vals = [np.hstack(vals[i]) for i in range(0, 5)]

plt.boxplot(vals[4])
plt.yscale("log")
plt.show()
