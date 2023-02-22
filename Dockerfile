FROM python:3.9-buster

LABEL org.opencontainers.image.source=https://github.com/nickthegroot/recipe-recommendation

WORKDIR /app

RUN python -m pip install -U pip setuptools wheel
COPY ./Makefile ./requirements-core.txt ./requirements.txt ./setup.py ./test_environment.py ./
RUN make requirements

# add source code
COPY ./data/test ./data/test
COPY ./src ./src

ENTRYPOINT [ "make" ]