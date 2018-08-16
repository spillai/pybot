FROM continuumio/miniconda3:latest

COPY . /source/pybot

WORKDIR /source/pybot

RUN conda create -yq -n pybot-runtime-env python=3.5
RUN conda install -c s_pillai pybot
