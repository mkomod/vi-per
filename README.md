## About

This repository implements the methods used in the paper "Logistic Variational Bayes Revisited" (2024)

TLDR: We introduce SOTA methods for variational logistic regression and GP classification

https://arxiv.org/abs/2406.00713

## Installing environment

The following command will install the environment for the project.

```bash
conda env create -f environment.yml
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

