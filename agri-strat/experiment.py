#!/usr/bin/env python

__author__ = "Juraj Micko"
__license__ = "MIT License"

if __name__ == '__main__':
    print("Importing modules...")

import argparse
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

import wandb
from models.base import BaseModelModule
from utils.callbacks.batch_counter import BatchCounterCallback
from utils.custom_progress_bar import CustomProgressBar
from utils.custom_wandb_logger import CustomWandbLogger
from utils.exception_tracker_callback import ExceptionTrackerCallback
from utils.label_encoder import LabelEncoder
from utils.medians_datamodule import MediansDataModule


def parse_arguments():
    # Parse user arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--job_type', type=str, default="train", required=False,
                        choices=['train', 'val'],
                        help='Type of job to perform. One of [\'train\', \'val\']')
    parser.add_argument('--tags', nargs="+", default=None, required=False,
                        help='Tags stored with the wandb job.')
    parser.add_argument('--notes', type=str, default=None, required=False,
                        help='Note stored with the wandb job.')

    parser.add_argument('--devtest', action='store_true', default=False, required=False,
                        help='Perform a dev test run with this model')

    parser.add_argument('--model', type=str, default="unet", required=False,
                        choices=['unet', 'convstar'],
                        help='Model to use. One of [\'unet\', \'convstar\']')
    parser.add_argument('--label_encoder_artifact', type=str, default="s4a_labels:latest", required=False,
                        help='Wandb artifact of type \'label_encoder\' that defines classes to be predicted.')
    parser.add_argument('--parcel_loss', action='store_true', default=False, required=False,
                        help='Use a loss function that takes into account parcel pixels only.')
    parser.add_argument('--weighted_loss', action='store_true', default=False, required=False,
                        help='Use a weighted loss function with precalculated weights per class. Default False.')
    parser.add_argument('--class_weights_weight', type=float, default=1.0, required=False,
                        help='Weight of the class weights in the loss function, interpolating between calculated '
                             'class weights and uniform weights. Default 1.0 = use calculated class weights.')
    parser.add_argument('--gradient_clip_val', type=float, default=10.0, required=False,
                        help='Gradient clipping value. Default 10.0')

    parser.add_argument('--medians_artifact', type=str, default="1MS_B02B03B04B08_61x61:latest", required=False,
                        help='Wandb artifact of type \'medians\' that references precomputed medians.')
    parser.add_argument('--medians_path', type=str, default=None, required=False,
                        help='Path to the directory with subdirectories of medians. Defaults to $MEDIANS_PATH or '
                             'dataset/medians.')
    parser.add_argument('--split_artifact', type=str, default="s4a_naive_split:latest", required=False,
                        help='Wandb artifact of type \'split\' that references the train/val/test splits.')
    parser.add_argument('--bins_range', type=int, nargs=2, default=[4, 9], required=False,
                        help='Specify to limit the range of the time bins (one-indexed, inclusive on both ends). '
                             'Default: [4, 9].')
    parser.add_argument('--skip_empty_subpatches', action='store_true', default=False, required=False,
                        help='Skip subpatches during training that have no pixels of interest (only relevant with '
                             'parcel_loss). Default False.')
    parser.add_argument('--shuffle_subpatches_within_patch', action='store_true', default=False, required=False,
                        help='Shuffle subpatches within each patch (only applies to training). Default False.')

    parser.add_argument('--num_epochs', type=int, default=10, required=False,
                        help='Number of epochs. Default 10')
    parser.add_argument('--batch_size', type=int, default=4, required=False,
                        help='If physical_batch_size=None, behaves as the conventional batch_size. Otherwise, '
                             'sets the effective batch size and must be smaller than or a multiply of '
                             'physical_batch_size. Default 4')
    parser.add_argument('--physical_batch_size', type=int, default=None, required=False,
                        help='If set and smaller than batch_size, sets the batch_size the model receives in each step.')
    parser.add_argument('--learning_rate', type=float, default=1e-1, required=False,
                        help='Starting learning rate. Default 1e-1')
    parser.add_argument('--requires_norm', action='store_true', default=False, required=False,
                        help='Normalize data to 0-1 range. Default False')
    parser.add_argument('--limit_train_batches', type=float, default=None, required=False,
                        help='Limit the number of batches to run during training (float = fraction, '
                             'int = num_batches). Default None')
    parser.add_argument('--limit_val_batches', type=float, default=None, required=False,
                        help='Limit the number of batches to run during validation (float = fraction, '
                             'int = num_batches). Default None')

    parser.add_argument('--deterministic', action='store_true', default=False, required=False,
                        help='Enforce reproducible results (except functions without a deterministic implementation). '
                             'Default False')
    parser.add_argument('--seed', type=int, default=0, required=False, )
    parser.add_argument('--shuffle_buffer_num_patches', type=int, default=100, required=False,
                        help='Size of the buffer for shuffling subpatches, given in number of patches. Default 100.')

    parser.add_argument('--num_workers', type=int, default=6, required=False,
                        help='Number of workers to work on dataloader. Default 6')
    parser.add_argument('--num_gpus', type=int, default=1, required=False,
                        help='Number of gpus to use (per node). Default 1')
    parser.add_argument('--num_nodes', type=int, default=1, required=False,
                        help='Number of nodes to use. Default 1')

    return parser.parse_args()


def get_config(args):
    exclude_keys = {"tags", "notes", "job_type"}
    config = {k: v for k, v in vars(args).items() if k not in exclude_keys}
    return config


def create_datamodule(config, label_encoder, calculated_batch_size):
    datamodule = MediansDataModule(
        medians_artifact=config["medians_artifact"],
        medians_path=config["medians_path"] or os.getenv("MEDIANS_PATH", "dataset/medians"),
        split_artifact=config["split_artifact"],
        bins_range=config["bins_range"],
        label_encoder=label_encoder,
        requires_norm=config["requires_norm"],
        batch_size=calculated_batch_size,
        num_workers=config["num_workers"],
        shuffle_buffer_num_patches=config["shuffle_buffer_num_patches"],
        skip_zero_label_subpatches=config["parcel_loss"] and config["skip_empty_subpatches"],
        # integer limits are passed directly to the trainer instead
        limit_train_batches=limit if (limit := config["limit_train_batches"]) % 1.0 != 0 else None,
        limit_val_batches=limit if (limit := config["limit_val_batches"]) % 1.0 != 0 else None,
        shuffle_subpatches_within_patch=config["shuffle_subpatches_within_patch"],
    )
    datamodule.prepare_data()
    datamodule.setup('fit')
    return datamodule


def create_model(config, label_encoder: LabelEncoder, datamodule: MediansDataModule):
    unsaved_params = dict(
        class_counts=datamodule.pixel_counts['train'],
        label_encoder=label_encoder,
        bands=datamodule.get_bands(),
        num_time_steps=config["bins_range"][1] - config["bins_range"][0] + 1,
        medians_metadata=datamodule.metadata,
    )
    if wandb.run.resumed:
        # Load the model from the latest checkpoint
        checkpoint_path = Path(wandb.run.dir) / "checkpoints" / "last.ckpt"
        if checkpoint_path.exists():
            print(f"Resuming model, loading from checkpoint {checkpoint_path}")
            return BaseModelModule.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                **unsaved_params,
            )
        else:
            print("Resuming a wandb run, but no checkpoint found. Creating a new model.")

    # Create a new model
    return BaseModelModule(
        weighted_loss=config["weighted_loss"],
        class_weights_weight=config["class_weights_weight"],
        model=config["model"],
        parcel_loss=config["parcel_loss"],
        num_layers=3,
        learning_rate=config["learning_rate"],
        **unsaved_params,
    )


def main():
    args = parse_arguments()

    print(f"Intializing wandb run... (cwd: {os.getcwd()})")
    with wandb.init(
            project="agri-strat",
            job_type=args.job_type,
            notes=args.notes,
            tags=args.tags,
            resume="allow",
            config=get_config(args),
            settings=wandb.Settings(job_name="train1"),
    ) as run:
        run.summary["node_name"] = os.environ.get('NODE_NAME', None)

        torch.set_float32_matmul_precision('medium')
        seed_everything(run.config["seed"], workers=True)

        if (run.config["physical_batch_size"] is not None
                and run.config["physical_batch_size"] < run.config["batch_size"]):
            assert run.config["batch_size"] % run.config["physical_batch_size"] == 0, \
                "batch_size must be divisible by physical_batch_size."
            multiply = run.config["batch_size"] // run.config["physical_batch_size"]
            assert multiply & (multiply - 1) == 0, \
                "batch_size must be a power of 2 when using physical_batch_size."
            calculated_batch_size, accumulate_grad_batches = run.config["physical_batch_size"], multiply
        else:
            calculated_batch_size, accumulate_grad_batches = run.config["batch_size"], 1

        print("Creating datamodule, model, trainer...")
        label_encoder = LabelEncoder(run.config["label_encoder_artifact"])
        datamodule = create_datamodule(run.config, label_encoder, calculated_batch_size)
        model = create_model(run.config, label_encoder, datamodule)

        callbacks = [
            BatchCounterCallback(datamodule),
            ExceptionTrackerCallback(),
            LearningRateMonitor(),
            ModelCheckpoint(
                dirpath=Path(wandb.run.dir) / "checkpoints",
                filename='ckpt_epoch={epoch:02d}',
                monitor=model.monitor_metric,
                save_last=True,
                save_top_k=-1,
                mode='max',
                auto_insert_metric_name=False,
            ),
            CustomProgressBar(refresh_rate=1 if run.config["devtest"] else 10),
        ]
        logger = CustomWandbLogger(
            experiment=run,
            log_model='all',
            checkpoint_name=f"model-{run.name}"
        )

        trainer = pl.Trainer(
            accelerator="auto",
            devices=run.config["num_gpus"],
            num_nodes=run.config["num_nodes"],
            max_epochs=run.config["num_epochs"],
            check_val_every_n_epoch=1,
            precision='32-true',
            callbacks=callbacks,
            logger=logger,
            gradient_clip_val=run.config["gradient_clip_val"],
            # For ensuring determinism with nll_loss2d_forward_out_cuda_template,
            # see https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/9
            deterministic="warn" if run.config["deterministic"] else None,
            benchmark=not run.config["deterministic"],
            fast_dev_run=run.config["devtest"],
            # fractional batch limits are passed directly to the data module instead
            limit_train_batches=int(limit) if (limit := run.config["limit_train_batches"]) % 1.0 == 0 else None,
            limit_val_batches=int(limit) if (limit := run.config["limit_val_batches"]) % 1.0 == 0 else None,
            num_sanity_val_steps=2,
            accumulate_grad_batches=accumulate_grad_batches,
            # profiler='simple',
        )

        print("Training...")
        trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
