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
ax[0].plot(ms, res0[1], color="tab:blue")
ax[0].plot(ms, res0[2], color="tab:green", linestyle="--")
ax[0].plot(ms, res0[0], color="tab:orange")
ax[0].grid(alpha=0.2)
ax[0].set_title("(a)  $\\tau = 2.0$", loc="left")
ax[0].set_xlabel("$\\vartheta$")
ax[0].legend(("Proposed bound", "Monte-Carlo Estimate", "Jaakkola and Jordan (2000)"))
ax[0].set_ylabel('Estimate of $E_{X}[\log(1 + \exp(X))]$')

ax[1].plot(ss, res1[1], color="tab:blue")
ax[1].plot(ss, res1[2], color="tab:green", linestyle="--")
ax[1].plot(ss, res1[0], color="tab:orange")
ax[1].grid(alpha=0.2)
ax[1].set_title("(b)  $\\vartheta = 1.0$", loc="left")
ax[1].set_xlabel("$\\tau$")
ax[1].set_ylabel('Estimate of $E_{X}[\log(1 + \exp(X))]$')

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
plt.savefig("/home/michael/proj/papers/logistic_vb/figures/error.pdf", bbox_inches="tight")



# ---------------------------
#       Figure 4
# ---------------------------
torch.manual_seed(1)
m = torch.linspace(-3, 3, 25)
s = torch.linspace(0.1, 3.0, 30)
m, s = torch.meshgrid(m, s)
m = m.reshape(-1)
s = s.reshape(-1)
grid = torch.stack([m, s], dim=1)

# plot the results
fig, ax = plt.subplots(2, 2, figsize=(15*1.2, 3.5*3.6))
ax = ax.reshape(-1)
lets = ["(a) 0.5%", "(b) 1%", "(c) 2.5%", "(d) 5%"]


for k, val in enumerate([0.005, 0.01, 0.025, 0.05]):
# for k, val in enumerate([0.005]):
    ls = []

    # compute the number of terms needed to get the error below 0.01
    for m, s in grid:
        l = 1
        res_true = mc_est(m, s, n_samples=2000000)
        while True:
            res = nb(m, s, l_max=l)
            if torch.abs((res_true - res) / res) < val:
                ls.append(l)
                print(f"m={m}, s={s}, l={l}")
                break
            l += 1

    # prepare results for plotting
    ls = torch.tensor(ls)
    ls = ls.reshape(25, 30)
    ls = ls.rot90(1)
    ls = ls.detach().numpy()

    ax[k].matshow(ls, cmap='viridis', interpolation='none', aspect="auto", vmin=0, vmax=17)
    for (i, j), z in np.ndenumerate(ls):
        ax[k].text(j, i, '{:d}'.format(z), ha='center', va='center', color="black", fontsize=6)
    ax[k].set_xticks([])
    ax[k].set_yticks([])
    # ax[k].gca().xaxis.tick_bottom()
    # put xaxis ticks on bottom
    ax[k].xaxis.set_ticks_position('bottom')
    ax[k].set_xticks(np.arange(0, 25, 4))
    ax[k].set_yticks(np.arange(0, 30, 5))
    ax[k].set_yticklabels(np.arange(3.0, 0.0, -0.5))
    ax[k].set_xticklabels(np.arange(-3, 4, 1))
    ax[k].set_xlabel("$\\vartheta$")
    ax[k].set_ylabel("$\\tau$")
    ax[k].set_title(lets[k], loc="left")

# add the colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
fig.colorbar(ax[0].images[0], cax=cbar_ax)

# plt.show()
# plt.show()

plt.savefig("/home/michael/proj/papers/logistic_vb/figures/l_terms.pdf", bbox_inches="tight")

