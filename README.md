# Recipe Recommendation

_The recommendation engine for home-cooked recipes_

## Description

This project was created for UCSD's DSC 180: Data Science Capstone. According to the university, the course:

> Span(s) the entire lifecycle, including assessing the problem, learning domain knowledge, collecting/cleaning data, creating a model, addressing ethical issues, designing the system, analyzing the output, and presenting the results.
>
> https://catalog.ucsd.edu/courses/DSC.html#dsc180b

## Installation

1. Install [poetry](https://python-poetry.org/docs/#installation)
2. Install all requirements using `poetry install`
3. Download the data from [Kaggle](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions) and unzip into `data/raw`
4. Clean the data by running `poetry run make_dataset`
5. Create a TigerGraph instance
6. Create the graph schema by running `data/schema.gsql` directly on the instance
7. Create a `.env` file using the `.env.sample` template, and fill with your instance's information.
8. Load the data into TigerGraph by running `poetry run sync_dataset`
