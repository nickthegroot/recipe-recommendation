FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# upgrade pip
RUN pip install --upgrade pip

# install poetry
RUN pip install poetry

# disable virtualenv for poetry
COPY ./pyproject.toml /app/pyproject.toml
RUN poetry config virtualenvs.create false

# install dependencies
RUN poetry install

# copy contents of project into docker
COPY ./model/ /app/model/

COPY ./pyproject.toml /app/pyproject.toml
COPY ./models/ /app/models/

COPY ./recipe_recommendation/util.py /app/recipe_recommendation/util.py
COPY ./recipe_recommendation/config.py /app/recipe_recommendation/config.py

COPY ./recipe_recommendation/core /app/recipe_recommendation/core
COPY ./recipe_recommendation/server /app/recipe_recommendation/server
COPY ./recipe_recommendation/model/ /app/recipe_recommendation/model


# set path to our python api file
ENV MODULE_NAME="recipe_recommendation.server.api"
ENV PYTHONPATH=/app/recipe_recommendation/