import torch
import torch.nn as nn

# generate data
n = 1000
p = 10
X = torch.randn(n, p)
b = torch.randn(p, 1)
intercept = torch.randn(1)
probs = X @ b + intercept + torch.randn(n, 1)
y = torch.bernoulli(torch.sigmoid(probs))

X_pred = torch.randn(n, p)
probs_pred = X_pred @ b + intercept + torch.randn(n, 1)
y_pred2 = torch.bernoulli(torch.sigmoid(probs_pred))

# define model
m = torch.randn(p, 1, requires_grad=True)
inter = torch.randn(1, requires_grad=True)

# define loss
def log_likelihood(m, inter, X, y):
    p = 1.0 + torch.exp(-X @ m - inter)
    return (y - 1).t() @ (X @ m + inter) - torch.sum(torch.log(p))

# define optimizer
lr = 0.01
n_epochs = 10000

for epoch in range(n_epochs):
    l = -log_likelihood(m, inter, X, y)
    l.backward()
    with torch.no_grad():
        m -= lr * m.grad
        inter -= lr * inter.grad
        m.grad.zero_()
        inter.grad.zero_()

torch.sum(torch.round(torch.sigmoid(X @ m + inter)) == y)

class classifiction_NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(classifiction_NN, self).__init__()
        self.linear1 = nn.Linear(input_dim, 10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3= nn.Linear(10, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x


model = classifiction_NN(p, 1)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

n_epochs = 10000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(loss.item())

sum(y == torch.round(model(X))) / 1000
sum(y_pred2 == torch.round(model(X_pred))) / 1000
sum(y_pred2 == torch.round(torch.sigmoid(X_pred @ m + inter))) / 1000



