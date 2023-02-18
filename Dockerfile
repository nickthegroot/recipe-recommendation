FROM ucsdets/scipy-ml-notebook:2023.1-stable

LABEL org.opencontainers.image.source=https://github.com/nickthegroot/recipe-recommendation

RUN python -m pip install -U pip setuptools wheel
RUN python -m pip install -r requirements.txt

# add source code
COPY ./data/test ./data/test
COPY ./Makefile Makefile
COPY ./recipe_recommendation ./recipe_recommendation
RUN poetry install --no-cache --only-root

ENTRYPOINT [ "make" ]