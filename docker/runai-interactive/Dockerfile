#ARG ROOT_CONTAINER=ubuntu:22.04
FROM jjurm/runai-job

EXPOSE 22

ENTRYPOINT ["tini", "-g", "--"]
CMD ["sudo", "/usr/sbin/sshd", "-D"]
