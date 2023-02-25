---
sidebar_position: 2
---

# Installation

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
   - You should see two files: `data/raw/interactions.csv` and `data/raw/recipes.csv`
3. Run `make data` to clean the data into its cleaned form.

## Running

All models can be trained using `python src/cli/train.py`.

- Run `python src/cli/train.py --help` for all configuration options (there's a lot!)
- In general, all models can be trained via `python src/cli/train.py fit`
  - For example, `HeteroLCN` is trained with `python src/cli/train.py fit --model HeteroLGN`
