FROM spillai/pybot-docker-base

COPY . /source/pybot

WORKDIR /source/pybot

# CHANGE THIS TO CONDA BUILD/DEVEL
# conda install -c s_pillai pybot
# conda create --name pybot -f conda_requirements.txt

RUN pip install -r requirements.txt && \
    python setup.py build
