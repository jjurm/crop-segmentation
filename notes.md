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

run docker image on kp machine for debugging

```bash
sudo docker run \
  -it --rm \
  --gpus all \
  --shm-size=24gb \
  --mount src=/home/jmicko/thesis-python/agri_strat/dataset,target=/workdir/dataset,type=bind \
  --mount src=/home/jmicko/thesis-python,target=/workdir/thesis-python,type=bind \
  --mount src=/home/jmicko/.config/wandb,target=/workdir/.config/wandb,type=bind \
  --mount src=/home/jmicko/.env,target=/workdir/.env,type=bind \
  --mount src=/home/jmicko/env.sh,target=/workdir/env.sh,type=bind \
  --hostname $(hostname)-debug \
  --env NODE_NAME=$(hostname)-debug \
  --env ENV_FILE=/workdir/.env \
  -p 2022:22 \
  jjurm/runai-python-job \
  wandb launch-agent
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
scheduler
```bash
runai submit \
  --image jjurm/runai-python-job \
  --working-dir /workdir \
  --cpu 1 \
  --node-type "CPU" \
  -e HOME=/workdir \
  -e ENV_FILE=/myhome/.env \
  --backoff-limit 0 \
  --preemptible \
  --name sched \
  -- wandb launch-agent -c /myhome/.config/wandb/launch-config-scheduler.yaml
```

create a job
```bash
wandb job create -p agri-strat -e jjurm --name split_data --entry-point agri_strat/split_data.py --git-hash $(git rev-parse HEAD) --runtime 3.10 git https://github.com/jjurm/master-thesis-code.git
```
```bash
wandb job create -p agri-strat -e jjurm --name experiment --entry-point agri_strat/experiment.py --git-hash $(git rev-parse HEAD) --runtime 3.10 git https://github.com/jjurm/master-thesis-code.git
wandb job create -p agri-strat -e jjurm --name train_test --entry-point "agri_strat/experiment.py --job_type train_test" --git-hash $(git rev-parse HEAD) --runtime 3.10 git https://github.com/jjurm/master-thesis-code.git
wandb job create -p agri-strat -e jjurm --name test --entry-point "agri_strat/experiment.py --job_type test" --git-hash $(git rev-parse HEAD) --runtime 3.10 git https://github.com/jjurm/master-thesis-code.git
```

launch the job
```bash
wandb launch -e jjurm -p agri-strat -q kp --job jjurm/agri-strat/split_data:latest --priority medium --config wandb_jobs/split_data.json
```
```bash
wandb launch -e jjurm -p agri-strat -q mixed --job jjurm/agri-strat/experiment:latest --priority medium --config wandb_jobs/experiment.json
```

split_artifacts: exp1_patches5000_strat_coco_split / s4a_temporal2_split / (temporal1)
model: unet/convstar
epochs: 40


## Sweep

```bash
wandb launch-sweep wandb_jobs/sweep2.yaml --queue kp --entity jjurm --project agri-strat
```


## S4A-Models: first reproduction

coco split data

```bash
python coco_data_split.py \
  --how stratified \
  --tiles 31TCJ 31TCL \
  --years 2019 \
  --seed 0 \
  --plot_distros \
  --prefix rep1-2tiles-e
```

calculate medians

```bash
python export_medians_multi.py \
  --prefix_coco rep1-2tiles-e \
  --group_freq 1MS \
  --bands B02 B03 B04 B08 \
  --output_size 61 61 \
  --out_path logs/medians/medians_1MS_B02B03B04B08 \
  --num_workers 32
```

compute class weights

```bash
python compute_class_weights.py \
  --coco_prefix rep1-2tiles-e \
  --saved_medians_path logs/medians/medians_1MS_B02B03B04B08 \
  --netcdf_path dataset/netcdf \
  --fixed_window
```

train

```bash
python pad_experiments.py --train \
  --model unet \
  --parcel_loss \
  --weighted_loss \
  --root_path_coco coco_files \
  --prefix_coco rep1-2tiles-e \
  --netcdf_path dataset \
  --prefix rep1-2tiles-e-run-a \
  --num_epochs 10 \
  --batch_size 32 \
  --bands B02 B03 B04 B08 \
  --img_size 61 61 \
  --requires_norm \
  --num_workers 6 \
  --num_gpus 1 \
  --saved_medians \
  --saved_medians_path logs/medians/medians_1MS_B02B03B04B08 \
  --fixed_window
```

tensorboard

```bash
tensorboard --bind_all --port 8888 --path_prefix /studentmichele-juraj/$JOB_NAME --logdir /mydata/studentmichele/juraj/thesis-python/S4A-Models/logs/unet/...run/run_timestamp
```


# KP workstation

## Install nvidia/cuda/cudnn

https://www.reddit.com/r/Fedora/comments/12xfowq/cuda_on_fedora_38_eta/
:

It takes a long time to compile GCC12, but CUDA Toolkit can be used this way in Fedora 38.

https://www.if-not-true-then-false.com/2023/fedora-build-gcc/

https://www.if-not-true-then-false.com/2018/install-nvidia-cuda-toolkit-on-fedora/#11-build-and-install-gcc-122

Also you can link gcc12 to CUDA bin directory to ease gcc version configuraition:

ln -s /usr/bin/gcc-12.2 /usr/local/cuda/bin/gcc

OR

Do with RPM Fusion (does much of the work for you)
https://rpmfusion.org/Configuration
https://rpmfusion.org/Howto/NVIDIA

CUDA:
```
sudo dnf install --allowerasing xorg-x11-drv-nvidia-cuda
```


## Ideas

- consider adding dropout (maybe only if the val/loss starts increasing)
- detect 'wrong weights' early (threshold on the accumulated loss after a couple batches)
- start off a pretrained model

- a compromise between iterative and map-style dataset:
  - have multiple workers return whole patches
  - combine into a single dataset
  - convert patches into subpatches
