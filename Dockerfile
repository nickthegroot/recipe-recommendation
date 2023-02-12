FROM ucsdets/scipy-ml-notebook:2023.1-stable

LABEL org.opencontainers.image.source=https://github.com/nickthegroot/recipe-recommendation

RUN pip install --upgrade pip
RUN pip install poetry

# disable virtualenv for poetry
RUN poetry config virtualenvs.create false

COPY ./poetry.lock ./pyproject.toml ./
RUN poetry install --no-cache

# add source code
COPY ./data/test ./data/test
COPY ./recipe_recommendation ./recipe_recommendation
RUN poetry install --no-cache --only-root

ENTRYPOINT [ "make" ]