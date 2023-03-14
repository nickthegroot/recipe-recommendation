<h1 align="center">
   <img src="reports/badges/ucsdseal.png" width=20% />
   <img src="reports/badges/tigergraph.png" width=20% />

Personalized Recipe Recommendation Using Heterogeneous Graphs

</h1>

**Authors**:

- Nicholas DeGroot (Halıcıoğlu Data Science Institute, UC San Diego)

## Description

This project was created for UCSD's DSC 180: Data Science Capstone. According to the university, the course:

> Span(s) the entire lifecycle, including assessing the problem, learning domain knowledge, collecting/cleaning data, creating a model, addressing ethical issues, designing the system, analyzing the output, and presenting the results.
>
> https://catalog.ucsd.edu/courses/DSC.html#dsc180b

## Getting Started

This project is configured with `devcontainer` support. This automatically creates a fully isolated environment with all required dependencies installed.

The easiest way to get started with `devcontainers` is through [GitHub Codespaces](https://github.com/features/codespaces).

1. Click [here](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=571806935) to create a new codespace on this repository.
   - Alternatively, this can be done through the `gh` CLI.
2. Configure the codespace to your liking. We recommend the 8-core machine.
3. Start the codespace and connect. It might take a minute to install all the dependencies. Grab a :coffee:!
4. Connect to the codespace through your preferred method (browser / VS Code).

## Downloading/Preparing the Data

1. Download the data by creating an Kaggle account and downloading the [`shuyangli94/food-com-recipes-and-user-interactions`](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) dataset.
2. Unzip the data into `data/raw`.
   - You should see a number of files, including `data/raw/interactions_train.csv` and `data/raw/RAW_recipes.csv`
3. Run `make data` to clean the data into its cleaned form.

## Running

All models can be trained using `python src/cli/train.py`.

- Run `python src/cli/train.py --help` for all configuration options
- In general, all models can be trained via `python src/cli/train.py --model {model}`
  - For example, `LightGCN` is trained with `python src/cli/train.py --model LightGCN`
