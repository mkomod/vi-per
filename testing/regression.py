# linear regression in pytorch

# generate data
n = 500
p = 10
X = torch.randn(n, p)
b = torch.randn(p, 1)
intercept = torch.randn(1)
y = X @ b + intercept + torch.randn(n, 1)

# define model
m = torch.randn(p, 1, requires_grad=True)
inter = torch.randn(1, requires_grad=True)

# define loss
def loss(m, inter, X, y):
    return torch.mean((y - X @ m - inter) ** 2)

# define optimizer
lr = 0.01
n_epochs = 1000

for epoch in range(n_epochs):
    l = loss(m, inter, X, y)
    l.backward()
    with torch.no_grad():
        m -= lr * m.grad
        inter -= lr * inter.grad
        m.grad.zero_()
        inter.grad.zero_()
