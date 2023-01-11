# Recipe Recommendation

_The recommendation engine for home-cooked recipes_

## Description

This project was created for UCSD's DSC 180: Data Science Capstone. According to the university, the course:

> Span(s) the entire lifecycle, including assessing the problem, learning domain knowledge, collecting/cleaning data, creating a model, addressing ethical issues, designing the system, analyzing the output, and presenting the results.
>
> https://catalog.ucsd.edu/courses/DSC.html#dsc180a

## Installation

If running on a CUDA-enabled GPU, install all dependencies packages:

```bash
poetry install --with gpu
```

Otherwise, install all dependency packages with:

```bash
poetry install --with cpu
```

## Project Organization

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a YYMMDD date (for ordering),
    │                         the creator's username, and a short `-` delimited description, e.g.
    │                         `221128-nickthegroot-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── recipe_recommendation                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes recipe_recommendation a Python module
    │   │
    │   ├── cli            <- Scripts used for training, validating, etc
    │   │   └── train.py
    │   │   └── validate.py
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── core           <- Classes that can be used by sub packages
    │   │   └── mixins     <- Useful mixins
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── base_model.py
    │   │   └── example_model.py

---

<small>
Project based on the <a href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a></small>
