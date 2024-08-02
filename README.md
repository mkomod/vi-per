## About

This repository implements the methods used in the paper "Logistic Variational Bayes Revisited" (2024)

TLDR: We introduce SOTA methods for variational logistic regression and GP classification

https://arxiv.org/abs/2406.00713

## Installing environment

The following command will install the environment for the project.

```bash
conda env create -f environment.yml
```


## Quick start GP Classification

If you want to do GP classfication see

https://github.com/mkomod/vi-per/blob/main/notebooks/gp_simulations.ipynb

```python
from src._97_gpytorch import LogisticGPVI

model = LogisticGPVI(y, X, n_inducing=50, n_iter=200, verbose=False)
model.fit()

y_pred = model.predict(X)
```

If you are familiar with Gpytorch the following class is an implementation of VI-PER.

```python
class LogitLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    has_analytic_marginal = False

    def __init__(self, l_max=12.0, ):
        self.l_max = l_max
        self.l = torch.arange(1.0, self.l_max*2, 1.0, requires_grad=False)
        return super().__init__()
 
    def forward(self, function_samples, *args, **kwargs):
        """ defines the liklihood function """
        output_probs = torch.sigmoid(function_samples)
        return torch.distributions.Bernoulli(probs=output_probs)

    @torch.jit.export
    def expected_log_prob(self, y, function_dist, *args, **kwargs):
        """ compute the expected log probability """
        M = function_dist.mean.view(-1, 1)
        S = function_dist.stddev.view(-1, 1)
        V = S**2

        M_S = M / S
        ML = M * self.l
        SL = S * self.l
        VL = 0.5 * V * (self.l ** 2)
        
        y_M = torch.dot(y, M.squeeze())
        normal_term = torch.sum(S / math.sqrt(2 * torch.pi) * torch.exp(-0.5 * M**2 / V) + M * ndtr(M_S))
        series_term = torch.sum(
            (-1.0)**(self.l - 1.0) / self.l * (
                torch.exp(ML + VL + log_ndtr(-M_S - SL)) + torch.exp(-ML + VL + log_ndtr(M_S - SL))
            )
        )

        return y_M - normal_term - series_term
```



## Project structure

The project is structured as follows:

```
.
├── README.md
├── data (NOT INCLUDED IN REPOSITORY AS NOT PUBLICLY AVAILABLE YET) 
├── environment.yml
├── notebooks
├── results
├── figures
├── scripts
└── src
    ├── __00_funcs.py
    ├── __01__data_generation.py
    ├── __02__method.py
    ├── __03__simulations.py
    ├── __04__application.py
    ├── __05__figures.py
    ├── __06__tables.py
    ├── __07__gp_simulations.py
    ├── __08__earthquake.py
    └── __97__gpytorch.py
```

The `data` folder contains the data used in the project. It is not included in the repository as it is not publicly available yet.

The `notebooks` folder contains the notebooks used for exploratory data analysis and for the generation of the figures for GP simulations and the application to soil liquefaction.

The `results` folder contains the results of the simulations and the applications.

The `scripts` folder contains the scripts used for the generation of the results.

The `src` folder contains the source code of the project.


## Reproducing results

The results for the following sections can be reproduced by running the scripts in the `scripts` folder:

- Section 3.1: `01-logistic_regression_simulations.sh`
- Section 3.2: `02-gaussian_process_example.sh`
- Section 4.1: `03-earthquake.sh`
- Section 4.2: `04-applications.sh`

The results will be saved to the `results` folder.

## Generating figures

The figures can be reproduced by running the `__05__figures.py` script in the `src` folder. Furthermore, the figures for the GP example can be reproduced by running the gp_simulations notebook in the `notebooks` folder. The figures for the earthquake application can be reproduced by running the soil_liquefaction notebook in the `notebooks` folder. The figures will be saved to the `figures` folder.

