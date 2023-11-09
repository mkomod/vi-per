mvn = dist.Normal(torch.zeros(1) + 2, torch.ones(1) * 2)
x = mvn.sample((5000, ))
torch.mean(torch.log1p(torch.exp(x)))

def approx_log1pexp(m, s):
    l = torch.arange(1.0, 5.0*2, 1.0)
    res= torch.sum(
        s / math.sqrt(2.0 * torch.pi) * torch.exp(-m**2 / (2.0 * s**2)) + \
        m * ndtr(m / s)
    ) + \
    torch.sum(
        (-1)**(l-1) / l * (
            torch.exp( m * l + 0.5 * s**2 * (l ** 2) + log_ndtr(-m / s - s * l)) + \
            torch.exp(-m * l + 0.5 * s**2 * (l ** 2) + log_ndtr( m / s - s * l))
        )
    )
    return res

approx_log1pexp(torch.zeros(1)+2, torch.ones(1)*2)

