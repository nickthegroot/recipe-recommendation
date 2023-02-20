FROM ucsdets/scipy-ml-notebook:2023.1-stable

LABEL org.opencontainers.image.source=https://github.com/nickthegroot/recipe-recommendation

RUN python -m pip install -U pip setuptools wheel
COPY ./Makefile ./requirements.txt ./setup.py ./test_environment.py ./
RUN make requirements

# add source code
COPY ./data/test ./data/test
COPY ./src ./src

ENTRYPOINT [ "make" ]