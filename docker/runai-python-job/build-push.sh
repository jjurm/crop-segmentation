set -e
docker build --platform linux/amd64 -t jjurm/runai-python-job .
docker push jjurm/runai-python-job
