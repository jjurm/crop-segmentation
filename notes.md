## Runai

Run the container

```bash
kubectl apply -f runai/jupyter.yaml
runai bash jupyter

su -c "JUPYTER_PORT=8888 tmux -L lab" jovyan

jupyter lab
```

## Python requirements

```bash
echo PACKAGE >> requirements.txt
pip install -r requirements.txt --upgrade
pip freeze > requirements-freeze.txt
```


## Train Wandb

Artifact path:
    split_rules -> `split_data.py` -> splits -> `compute_medians.py` -> medians[] -> `experiment.py` -> model

generate a train/val/test split

```bash
wandb artifact put \
  --type split_rules \
  --name s4a_temporal1 \
  dataset/split_rules/s4a_temporal1.yaml

python split_data.py \
  --split_rules_artifact s4a_temporal1:latest \
  --netcdf_path ../data/S4A/data \
  --seed 0 \
  --shuffle
```

compute medians (and class counts)

```bash
python compute_medians.py \
  --splits_artifact s4a_temporal1_split:latest \
  --netcdf_path ../data/S4A/data \
  --group_freq 1MS \
  --bands B02 B03 B04 B08 \
  --output_size 61 61 \
  --num_workers 32
```

```bash
python experiment.py \
  --job_type train \
  --model unet \
  --parcel_loss \
  --weighted_loss \
  --train_medians_artifact s4a_temporal1_medians_84b6_train:latest \
  --val_medians_artifact s4a_temporal1_medians_84b6_val:latest \
  --bins_range 4 9 \
  --num_epochs 40 \
  --batch_size 32 \
  --lr 1e-1 \
  --requires_norm \
  --num_workers 16 \
  --cached_dataset

  #--parcel_loss \
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

ln -s /usr/bin/gcc-12.2/usr/local/cuda/bin/gcc

OR

Do with RPM Fusion (does much of the work for you)
https://rpmfusion.org/Configuration
https://rpmfusion.org/Howto/NVIDIA
