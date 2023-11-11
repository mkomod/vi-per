import torch
import torch.distributions as dist


def generate_data(n, p, b = torch.tensor([1., 0.5, -1., -0.5],
    dtype=torch.float64), seed=1, dgp=0):
    """
    Generate data for a logistic regression model
    :param n: number of samples
    :param p: number of features
    :return: data
    """
    # Set seed
    torch.manual_seed(seed)

    # Generate data
    if dgp == 0:
        X = torch.randn(n, p, dtype=torch.double)
    elif dgp == 1:
        # S_{ij} = 0.6^{abs(i-j)}
        S = torch.zeros(p, p, dtype=torch.double)
        for i in range(p):
            for j in range(p):
                S[i, j] = 0.3 ** torch.abs(torch.tensor(i - j, dtype=torch.double))
        mvn = dist.MultivariateNormal(torch.zeros(p, dtype=torch.double), S)
        X = mvn.sample((n,))
    elif dgp == 2:
        # U = torch.rand(p, p, dtype=torch.double)
        # S = torch.matmul(U, U.t())

        # generate a cov matrix
        df = p + 5
        wishart = dist.Wishart(df, covariance_matrix=torch.eye(p))
        S = wishart.sample((1,)).squeeze() / df
        S = S.type(torch.double)

        mvn = dist.MultivariateNormal(torch.zeros(p, dtype=torch.double), S)
        X = mvn.sample((n,))

    X = X.type(torch.double)

    if p != 4:
        b = torch.rand(p, dtype=torch.double)
    
    # sample b_j from U[-2, -0.2] and [0.2, 2]
    b = torch.zeros(p)
    for i in range(p):
        if torch.rand(1) < 0.5:
            b[i] = torch.rand(1) * -1.8 - 0.2
        else:
            b[i] = torch.rand(1) * 1.8 + 0.2
    b = b.type(torch.double)
    
    # Generate labels
    p = torch.sigmoid(X @ b)
    y = torch.bernoulli(p)
    y = y.type(torch.double)
    
    return {"X": X, "y": y, "b": b}

