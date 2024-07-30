This package contains the main code for running the experiments, including the
data processing, model training and evaluation.

Experiments are run using the Weight and Biases platform: all the runs are
logged and can be viewed in
the [project dashboard](https://wandb.ai/juraj_micko/agri_strat).

## Entry scripts

The following scripts are the main entrypoints for running the experiments.
Most of the scripts operate on the [Weights and Biases](https://wandb.ai/)
platform, so the user needs to have an account and be logged in to run them.
Additionally, scripts require certain environment variables to be set, which are
described in the `.env.template` file. Each script has a help message containing
information about the required and supported arguments. We list the scripts in
the order they are typically used.

- `download_dem.py` - Script for downloading the DEM data from the Copernicus
  DEM service.
- `compute_medians.py` - Script for computing the aggregated median pixel values
  of the Sentinel-2 bands for the training data.
- `split_data.py` - Script for splitting the data into train, val and test sets,
  based on a specification of split rules.
- `swap_train_val_split.py` - Script for swapping the train and val sets of an
  existing data split.
- `experiment.py` - Script for training and evaluating the models. Supports
  training from scratch, continuing a run, loading a model from a checkpoint,
  evaluating a model on the val and test sets and exporting various metrics and
  aggregated statistics.
