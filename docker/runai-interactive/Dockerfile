#ARG ROOT_CONTAINER=ubuntu:22.04
ARG ROOT_CONTAINER=jjurm/runai-job
FROM $ROOT_CONTAINER

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Set up SSH
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    openssh-server

EXPOSE 22

# Expose Jupyter port
ENV JUPYTER_PORT=8888
EXPOSE $JUPYTER_PORT

ENTRYPOINT ["tini", "-g", "--"]
CMD ["/usr/sbin/sshd", "-D"]
