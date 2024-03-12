set -e
docker build --platform linux/amd64 -t jjurm/runai-job .
docker push jjurm/runai-job
