#ARG ROOT_CONTAINER=ubuntu:22.04
ARG ROOT_CONTAINER=jjurm/runai-job
FROM $ROOT_CONTAINER

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

EXPOSE 22

ENTRYPOINT ["tini", "-g", "--"]
CMD ["/usr/sbin/sshd", "-D"]
