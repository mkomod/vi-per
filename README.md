
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

