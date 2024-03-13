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

import config as experiment_config
import wandb
from models.base import BaseModelModule
from utils.custom_progress_bar import CustomProgressBar
from utils.custom_wandb_logger import CustomWandbLogger
from utils.exception_tracker_callback import ExceptionTrackerCallback
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
    parser.add_argument('--limit_batches', type=int, nargs="+", default=None, required=False,
                        help='Limit the number of batches to run during training and validation. Default None')

    parser.add_argument('--model', type=str, required=True,
                        choices=['unet', 'convstar'],
                        help='Model to use. One of [\'unet\', \'convstar\']')

    parser.add_argument('--parcel_loss', action='store_true', default=False, required=False,
                        help='Use a loss function that takes into account parcel pixels only.')
    parser.add_argument('--weighted_loss', action='store_true', default=False, required=False,
                        help='Use a weighted loss function with precalculated weights per class. Default False.')

    parser.add_argument('--medians_artifact', type=str, required=True,
                        help='Wandb artifact of type \'medians\' that references precomputed medians.')
    parser.add_argument('--medians_path', type=str, default='dataset/medians', required=False,
                        help='Path to the directory with subdirectories of medians. Default dataset/medians')
    parser.add_argument('--split_artifact', type=str, required=True,
                        help='Wandb artifact of type \'split\' that references the train/val/test splits.')
    parser.add_argument('--bins_range', type=int, nargs=2, default=[4, 9], required=False,
                        help='Specify to limit the range of the time bins (one-indexed, inclusive on both ends). '
                             'Default: [4, 9].')

    parser.add_argument('--num_epochs', type=int, default=10, required=False,
                        help='Number of epochs. Default 10')
    parser.add_argument('--batch_size', type=int, default=4, required=False,
                        help='The batch size. Default 4')
    parser.add_argument('--lr', type=float, default=1e-1, required=False,
                        help='Starting learning rate. Default 1e-1')
    parser.add_argument('--requires_norm', action='store_true', default=False, required=False,
                        help='Normalize data to 0-1 range. Default False')

    parser.add_argument('--deterministic', action='store_true', default=False, required=False,
                        help='Enforce reproducible results (except functions without a deterministic implementation). '
                             'Default False')
    parser.add_argument('--seed', type=int, default=0, required=False, )

    parser.add_argument('--num_workers', type=int, default=6, required=False,
                        help='Number of workers to work on dataloader. Default 6')
    parser.add_argument('--cache_dataset', action='store_true', default=False, required=False,
                        help='Cache the dataset in memory during the first epoch. Default False.')
    parser.add_argument('--num_gpus', type=int, default=1, required=False,
                        help='Number of gpus to use (per node). Default 1')
    parser.add_argument('--num_nodes', type=int, default=1, required=False,
                        help='Number of nodes to use. Default 1')

    return parser.parse_args()


def get_config(args):
    config = {
        'model': args.model,
        'devtest': args.devtest,
        'limit_train_batches': args.limit_batches[0] if args.limit_batches and args.limit_batches[0] > 0 else None,
        'limit_val_batches': args.limit_batches[1] if args.limit_batches and len(args.limit_batches) >= 2 and
                                                      args.limit_batches[1] > 0 else None,

        'medians_artifact': args.medians_artifact,
        'medians_path': args.medians_path,
        'split_artifact': args.split_artifact,
        'bins_range': args.bins_range,

        'parcel_loss': args.parcel_loss,
        'monitor_metric': 'val/f1w_parcel' if args.parcel_loss else 'val/f1w',
        'weighted_loss': args.weighted_loss,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.lr,
        'requires_norm': args.requires_norm,

        'deterministic': args.deterministic,
        'seed': args.seed,

        'num_workers': args.num_workers,
        'cache_dataset': args.cache_dataset,
        'num_gpus': args.num_gpus,
        'num_nodes': args.num_nodes,
        'node_name': os.environ.get('NODE_NAME', None),
    }
    return config


def get_tags(args, config):
    tags = args.tags
    if args.devtest or args.limit_batches:
        tags.append("devtest")
    return tags


def create_datamodule(config):
    datamodule = MediansDataModule(
        medians_artifact=config["medians_artifact"],
        medians_path=config["medians_path"],
        split_artifact=config["split_artifact"],
        bins_range=config["bins_range"],
        linear_encoder=experiment_config.LINEAR_ENCODER,
        requires_norm=config["requires_norm"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        cache_dataset=config["cache_dataset"],
    )
    datamodule.prepare_data()
    datamodule.setup('fit')
    return datamodule


def create_model(config, datamodule):
    unsaved_params = dict(
        class_counts=datamodule.pixel_counts['train'],
        linear_encoder=experiment_config.LINEAR_ENCODER,
        bands=datamodule.get_bands(),
        num_time_steps=config["bins_range"][1] - config["bins_range"][0] + 1,
        medians_metadata=datamodule.metadata,
    )
    if wandb.run.resumed:
        # Load the model from the latest checkpoint
        checkpoint_path = Path(wandb.run.dir) / "checkpoints" / "last.ckpt"
        print(f"Resuming model, loading from checkpoint {checkpoint_path}")
        return BaseModelModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            **unsaved_params,
        )
    else:
        # Create a new model
        return BaseModelModule(
            weighted_loss=config["weighted_loss"],
            model=config["model"],
            parcel_loss=config["parcel_loss"],
            monitor_metric=config["monitor_metric"],
            num_layers=3,
            learning_rate=config["learning_rate"],
            **unsaved_params,
        )


def main():
    args = parse_arguments()

    print("Intializing wandb run...")
    with wandb.init(
            project="agri-strat",
            job_type=args.job_type,
            notes=args.notes,
            tags=args.tags,
            resume="allow",
            config=get_config(args),
            settings=wandb.Settings(job_name="train1"),
    ) as run:
        torch.set_float32_matmul_precision('medium')
        seed_everything(run.config["seed"], workers=True)

        print("Creating datamodule, model, trainer...")
        datamodule = create_datamodule(run.config)
        model = create_model(run.config, datamodule)

        callbacks = [
            ExceptionTrackerCallback(),
            LearningRateMonitor(),
            ModelCheckpoint(
                dirpath=Path(wandb.run.dir) / "checkpoints",
                filename='ckpt_epoch={epoch:02d}',
                monitor=run.config["monitor_metric"],
                save_last='link',
                save_top_k=-1,
                mode='max',
                auto_insert_metric_name=False,
            ),
            CustomProgressBar(),
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
            gradient_clip_val=10.0,
            # For ensuring determinism with nll_loss2d_forward_out_cuda_template,
            # see https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/9
            deterministic="warn" if run.config["deterministic"] else None,
            benchmark=not run.config["deterministic"],
            fast_dev_run=run.config["devtest"],
            limit_train_batches=run.config["limit_train_batches"],
            limit_val_batches=run.config["limit_val_batches"],
            num_sanity_val_steps=2,
            # profiler='simple',
        )

        print("Training...")
        trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
