#ARG ROOT_CONTAINER=ubuntu:22.04
ARG ROOT_CONTAINER=nvidia/cuda:12.3.1-devel-ubuntu22.04

FROM $ROOT_CONTAINER

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    bzip2 \
    ca-certificates \
    locales \
    sudo \
    tini \
    wget \
    fonts-liberation \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

RUN apt-get install --yes --no-install-recommends \
    pandoc \
    curl \
    git \
    tmux \
    nano-tiny \
    unzip \
    vim-tiny \
    openssh-server \
    openssh-client \
    less \
    && apt-get clean && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/run/sshd && \
    update-alternatives --install /usr/bin/nano nano /bin/nano-tiny 10

ARG HOME="/myhome"
ENV SHELL=/bin/bash \
    NB_USER="${NB_USER}" \
    NB_UID=${NB_UID} \
    NB_GID=${NB_GID} \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    HOME="${HOME}"

RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc

ARG NB_PASSWORD="jupyter"
RUN sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers && \
    echo "%sudo ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    useradd --home-dir "${HOME}" --no-create-home --shell /bin/bash --uid "${NB_UID}" --no-user-group "${NB_USER}" && \
    chmod g+w /etc/passwd && \
    adduser "${NB_USER}" sudo && \
    echo "${NB_USER}:${NB_PASSWORD}" | chpasswd

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod a+x /entrypoint.sh

SHELL ["/bin/bash", "-o", "pipefail", "-l", "-c"]

# Expose Jupyter port
ENV JUPYTER_PORT=8888
EXPOSE $JUPYTER_PORT

USER $NB_USER

WORKDIR "$HOME"
ENTRYPOINT ["tini", "-g", "--", "/bin/bash", "/entrypoint.sh"]
CMD ["/bin/bash"]
