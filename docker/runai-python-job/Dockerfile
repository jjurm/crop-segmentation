ARG ROOT_CONTAINER=jjurm/runai-job
FROM $ROOT_CONTAINER

USER root

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    python3.10 python3.10-dev \
    libcurl4-gnutls-dev \
    libgnutls28-dev \
    libomp-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/* && \
    sudo ln -s /usr/bin/python3.10 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    pip install --upgrade pip


USER $NB_USER

COPY requirements.txt /tmp/requirements.txt
RUN pip install --timeout 3600 -r /tmp/requirements.txt

ENV IMAGE_HAS_PYTHON=true
