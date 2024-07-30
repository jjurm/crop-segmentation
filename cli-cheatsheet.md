# CLI Commands Reference

This document provides personal-use CLI commands and arguments, with no warranties or guarantees.

## Runai

Run the container

```bash
kubectl apply -f runai/jupyter.yaml
runai bash jupyter
```

## Python requirements

```bash
echo PACKAGE >> requirements.txt
pip install -r requirements.txt --upgrade
pip freeze > requirements-freeze.txt
```


## Train Wandb

generate a train/val/test split

```bash
wandb artifact put \
  --type split_rules \
  --name s4a_temporal1 \
  dataset/split_rules/s4a_temporal1.yaml

python split_data.py \
  --split_rules_artifact s4a_temporal1:latest \
  --netcdf_path dataset/netcdf \
  --seed 0 \
  --shuffle \
  --elevation
  
# Or generate a split from train/val/test COCO files
python split_data.py \
  --coco_path_prefix dataset/coco_files/exp1_patches5000_strat_coco \
  --netcdf_path dataset/netcdf
```

compute medians (and class counts)

```bash
python compute_medians.py \
  --netcdf_path dataset/netcdf \
  --medians_path dataset/medians \
  --group_freq 1MS \
  --bands B02 B03 B04 B08 \
  --output_size 61 61 \
  --num_workers 32
```

upload label_encoder artifact

```bash
wandb artifact put \
  --type label_encoder \
  --name s4a_labels \
  agri_strat/dataset/label_encoder/s4a_labels.yaml
```

training experiment

```bash
python experiment.py \
  --job_type train \
  --model unet \
  --parcel_loss \
  --weighted_loss \
  --medians_artifact 1MS_B02B03B04B08_61x61:latest \
  --medians_path dataset/medians \
  --split_artifact exp1_patches5000_strat_coco_split:latest \
  --bins_range 4 9 \
  --num_epochs 40 \
  --batch_size 32 \
  --learning_rate 1e-1 \
  --requires_norm \
  --num_workers 8 \
  --shuffle_buffer_num_patches 500 \
  --skip_empty_subpatches
```


## Launch experiments

create a queue
```python
import wandb
wandb.Api().create_run_queue(name="mixed", type="local-process", prioritization_mode="V0")
```

start an agent

kp machine:
```bash
sudo docker run \
  -it --rm \
  --gpus all \
  --env-file /home/jmicko/.env \
  --shm-size=24gb \
  --mount src=/home/jmicko/thesis-python/agri_strat/dataset,target=/workdir/dataset,type=bind \
  --mount src=/home/jmicko/.config/wandb,target=/workdir/.config/wandb,type=bind \
  --hostname $(hostname)-docker \
  --env NODE_NAME=$(hostname)-docker \
  -p 2022:22 \
  jjurm/runai-python-job \
  wandb launch-agent
# -c /workdir/.config/wandb/launch-config-2.yaml
```

on runai:
```bash
runai submit \
  --image jjurm/runai-python-job \
  --working-dir /workdir \
  --cpu 4 \
  --memory 18G \
  --node-type "A100" \
  --gpu 0.1 \
  --large-shm \
  -e HOME=/workdir \
  -e ENV_FILE=/myhome/.env \
  --backoff-limit 0 \
  --preemptible \
  --name wla1 \
  -- wandb launch-agent -c /myhome/.config/wandb/launch-config.yaml
```

create a job
```bash
wandb job create -p agri-strat -e jjurm --name split_data --entry-point agri_strat/split_data.py --git-hash $(git rev-parse HEAD) --runtime 3.10 git https://github.com/jjurm/crop-segmentation.git
```
```bash
wandb job create -p agri-strat -e jjurm --name experiment --entry-point agri_strat/experiment.py --git-hash $(git rev-parse HEAD) --runtime 3.10 git https://github.com/jjurm/crop-segmentation.git
wandb job create -p agri-strat -e jjurm --name train_test --entry-point "agri_strat/experiment.py --job_type train_test" --git-hash $(git rev-parse HEAD) --runtime 3.10 git https://github.com/jjurm/crop-segmentation.git
wandb job create -p agri-strat -e jjurm --name test --entry-point "agri_strat/experiment.py --job_type test" --git-hash $(git rev-parse HEAD) --runtime 3.10 git https://github.com/jjurm/crop-segmentation.git
```

launch the job
```bash
wandb launch -e jjurm -p agri-strat -q kp --job jjurm/agri-strat/split_data:latest --priority high --config wandb_jobs/split_data.json
```
```bash
wandb launch -e jjurm -p agri-strat -q mixed --job jjurm/agri-strat/experiment:latest --priority medium --config wandb_jobs/experiment.json
wandb launch -e jjurm -p agri-strat -q kp --job jjurm/agri-strat/train_test:latest --priority high --config wandb_jobs/experiments/active_sampling/unet_bl2_loss.json
```


## Sweep

```bash
wandb launch-sweep --queue kp --entity jjurm --project agri-strat wandb_jobs/experiments/active_sampling/
```
