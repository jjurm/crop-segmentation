# Crop segmentation with active sampling

This repository contains the code for the master thesis "Optimizing large-scale satellite-based crop segmentation with deep learning and active sampling" by Juraj Mičko.


## Code structure

- `agri_strat` - main code and entrypoints for running the experiments, including the data processing, model training and evaluation
- `docker` - Dockerfiles for building the image for running the experiments
- `qgis` - helper projects and assets for exploring the data in QGIS
- `runai` - scripts and pod templates for running the experiments on a Run:AI cluster
- `wandb_jobs` - configuration files for running jobs and sweeps, sweeps and replicating the results from the thesis

See [agri_strat/README.md](agri_strat/README.md) for more information about the individual entry point scripts and details on the data processing pipeline.


## Running the experiments

All experiments rely on the [Weights and Biases](https://wandb.ai/) platform for logging the runs and artifacts.
In this section we describe a common approach to running the experiments.

### Download the dataset

Before training takes place, the [Sen4AgriNet dataset](https://huggingface.co/datasets/paren8esis/S4A) needs to be downloaded.

```bash
git lfs install
git clone https://huggingface.co/datasets/paren8esis/S4A
```

Additionally, elevation data (Digital Elevation Model, DEM) can be made accessible in the dataset split creation phase to annotate patches with elevation information (e.g. to later compute statistics or create figures).

```bash
python download_dem.py --xy_lims -5 9.5 40 51
```

### Preprocess the dataset

We compute the monthly medians of the Sentinel-2 bands for the training
data using `compute_medians.py`.

```bash
python compute_medians.py \
  --netcdf_path dataset/netcdf \
  --medians_path dataset/medians \
  --group_freq 1MS \
  --bands B02 B03 B04 B08 \
  --output_size 61 61 \
  --num_workers 32
```

### Create a `split_rules` artifact

A `split_rules` artifact is a yaml file that specifies the rules for splitting the available data into train, val and test sets.
This artifact is just a specification of how to create the actual split, and so it is not tied to any specific data.
Split rules used in our work are located in the `agri_strat/dataset/split_rules` directory.

```bash
wandb artifact put \
  --type split_rules \
  --name s4a_naive \
  dataset/split_rules/s4a_naive.yaml
```

Optionally, we can swap the train and val sets of an existing split (e.g. to
train a holdout model for the RHO-Loss active sampling strategy)
using `swap_train_val_split.py`.

```bash
python swap_train_val_split.py \
  --split_artifact s4a_naive:latest
```

### Create a `split` artifact

The split artifact contains a list of patches for each of the train, val and test sets.
These splits are independent from the precomputed medians, and can be combined with them arbitrarily (the splits only contain information about which patches belong to each set).
A `split` artifact is created based on a `split_rules` artifact, and accesses the downloaded Sen4AgriNet dataset to compute statistics about the patches, assign them to the sets, and save the resulting split to Weights and Biases along with the statistics.

```bash
python split_data.py \
  --split_rules_artifact s4a_temporal1:latest \
  --netcdf_path agri_strat/dataset/netcdf \
  --seed 0 \
  --shuffle \
  --elevation
```

However, instead of generating a split according to one of the split rules, a COCO-style file can be used to generate the split, which is useful when replicating the splits used in the [Sen4AgriNet paper](https://github.com/Orion-AI-Lab/S4A-Models).

```bash
python split_data.py \
--coco_path_prefix agri_strat/dataset/coco_files/exp1_patches5000_strat_coco \
--netcdf_path agri_strat/dataset/netcdf
```

### Create a `label_encoder` artifact

The `label_encoder` artifact contains a definition of the set of labels used in the dataset.
Each label has a name and can be mapped to one or more raw class IDs in the dataset.
Label encoders used in our work are located in the `agri_strat/dataset/label_encoder` directory.

```bash
wandb artifact put \
  --type label_encoder \
  --name s4a_labels \
  agri_strat/dataset/label_encoder/s4a_labels.yaml
```

### Training a model (locally)

For a simple way to train a model locally, we can launch the `experiment.py` script directly. For example:

```bash
cd agri_strat
python experiment.py \
  --job_type train \
  --model unet \
  --parcel_loss \
  --weighted_loss \
  --medians_artifact 1MS_B02B03B04B08_61x61:latest \
  --medians_path dataset/medians \
  --split_artifact s4a_naive:latest \
  --bins_range 4 9 \
  --num_epochs 40 \
  --batch_size 32 \
  --learning_rate 1e-1 \
  --requires_norm \
  --num_workers 8 \
  --shuffle_buffer_num_patches 500 \
  --skip_empty_subpatches
```

See `python experiment.py --help` for all available options.

### Training a model (W&B Launch job)

Another option to run the training is to run an agent connected to a queue within the Weights and Biases platform, and then submit a [Launch job](https://docs.wandb.ai/guides/launch) to the queue.
Create a queue with the following:

```python
import wandb
wandb.Api().create_run_queue(name="mixed", type="local-process", prioritization_mode="V0")
```

#### Start an agent

```bash
docker run \
  -it --rm \
  --gpus all \
  --env-file <your .env> \
  jjurm/runai-python-job \
  wandb launch-agent
```

Don't forget that you might need to configure W&B within your system and `--mount` any necessary directories into the container such as the dataset.

#### Create a job

This step involves registering an artifact with W&B that contains a link to a specific commit within the repository and the entrypoint script to run the experiment.

```bash
# For split_data.py
wandb job create -p agri-strat -e jjurm --name split_data --entry-point agri_strat/split_data.py --git-hash $(git rev-parse HEAD) --runtime 3.10 git https://github.com/jjurm/crop-segmentation.git

# For experiment.py
wandb job create -p agri-strat -e jjurm --name experiment --entry-point agri_strat/experiment.py --git-hash $(git rev-parse HEAD) --runtime 3.10 git https://github.com/jjurm/crop-segmentation.git

# For experiment.py --job_type train_test
wandb job create -p agri-strat -e jjurm --name train_test --entry-point "agri_strat/experiment.py --job_type train_test" --git-hash $(git rev-parse HEAD) --runtime 3.10 git https://github.com/jjurm/crop-segmentation.git

# For experiment.py --job_type test
wandb job create -p agri-strat -e jjurm --name test --entry-point "agri_strat/experiment.py --job_type test" --git-hash $(git rev-parse HEAD) --runtime 3.10 git https://github.com/jjurm/crop-segmentation.git
```

#### Prepare a run configuration

See `wandb_jobs/experiment.json` for an example of a run configuration file that can be used to run `experiment.py`.

#### Launch a run

```bash
wandb launch -e jjurm -p agri-strat -q kp --job jjurm/agri-strat/train_test:latest --priority high --config wandb_jobs/experiments/active_sampling/unet_bl2_loss.json
```

#### Launch a sweep of runs

Similar to a single run, a yaml file can describe a sweep of runs to be launched. See `wandb_jobs/experiments/replicate_sen4agrinet/sweep_unet.yaml` for an example of a sweep configuration file.

```bash
wandb launch-sweep --queue kp --entity jjurm --project agri-strat wandb_jobs/experiments/replicate_sen4agrinet/sweep_unet.yaml
```

The `wandb_jobs/experiments` directory contains sweep configuration files to reproduce the experiments from the thesis.


## Installing the `agri_strat` Python package

The python project is structured as a package and can be installed so that the code can be imported as the `agri_strat` library. This is useful for reusing the code in other projects such as for generating figures.
The package can be installed by running the following command in the root of the repository:

```bash
pip install --editable .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
