FROM continuumio/miniconda3:latest

RUN conda install -y pip requests conda-build anaconda-client
RUN conda create -q -n pybot-build-env python=3.5
ENV PATH /opt/conda/envs/pybot-build-env/bin:$PATH
RUN conda config --add channels menpo

COPY . /source/pybot
WORKDIR /source/pybot

RUN apt-get update && apt-get install -y build-essential
RUN conda build tools/conda.recipe
RUN conda install --use-local pybot
