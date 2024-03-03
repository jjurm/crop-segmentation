#! /usr/bin/env python

__author__ = "Juraj Micko"
__license__ = "MIT License"

if __name__ == '__main__':
    print("Importing modules...")

import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import config as experiment_config
import wandb
from models.base import BaseModelModule
from utils.custom_progress_bar import CustomProgressBar
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

    parser.add_argument('--train_medians_artifact', type=str, default='logs/medians', required=False,
                        help='Wandb artifact of type \'medians\' that references precomputed medians for training.')
    parser.add_argument('--val_medians_artifact', type=str, default='logs/medians', required=False,
                        help='Wandb artifact of type \'medians\' that references precomputed medians for validation.')
    parser.add_argument('--test_medians_artifact', type=str, default='logs/medians', required=False,
                        help='Wandb artifact of type \'medians\' that references precomputed medians for test.')
    parser.add_argument('--bins_range', type=int, nargs=2, default=[4, 9], required=True,
                        help='Specify to limit the range of the time bins (one-indexed, inclusive on both ends). Default: [4, 9].')

    parser.add_argument('--num_epochs', type=int, default=10, required=False,
                        help='Number of epochs. Default 10')
    parser.add_argument('--batch_size', type=int, default=4, required=False,
                        help='The batch size. Default 4')
    parser.add_argument('--lr', type=float, default=1e-1, required=False,
                        help='Starting learning rate. Default 1e-1')
    parser.add_argument('--requires_norm', action='store_true', default=False, required=False,
                        help='Normalize data to 0-1 range. Default False')
    parser.add_argument('--deterministic', action='store_true', default=False, required=False,
                        help='Enforce reproducible results. Default False')

    parser.add_argument('--num_workers', type=int, default=6, required=False,
                        help='Number of workers to work on dataloader. Default 6')
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

        'medians_artifacts': {
            'train': args.train_medians_artifact,
            'val': args.val_medians_artifact,
            'test': args.test_medians_artifact,
        },
        'bins_range': args.bins_range,

        'parcel_loss': args.parcel_loss,
        'monitor_metric': 'val/f1w_parcel' if args.parcel_loss else 'val/f1w',
        'weighted_loss': args.weighted_loss,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.lr,
        'requires_norm': args.requires_norm,
        'deterministic': args.deterministic,

        'num_workers': args.num_workers,
        'num_gpus': args.num_gpus,
        'num_nodes': args.num_nodes,
    }
    return config


def create_datamodule(config):
    datamodule = MediansDataModule(
        medians_artifacts=config["medians_artifacts"],
        bins_range=config["bins_range"],
        linear_encoder=experiment_config.LINEAR_ENCODER,
        requires_norm=config["requires_norm"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    datamodule.prepare_data()
    datamodule.setup('fit')
    return datamodule


def create_model(config, datamodule):
    return BaseModelModule(
        model=config["model"],
        linear_encoder=experiment_config.LINEAR_ENCODER,
        parcel_loss=config["parcel_loss"],
        monitor_metric=config["monitor_metric"],
        class_weights=experiment_config.CLASS_WEIGHTS,
        num_layers=3,
        num_bands=datamodule.get_num_bands(),
        num_time_steps=config["bins_range"][1] - config["bins_range"][0] + 1,
        learning_rate=config["learning_rate"],
    )


def main():
    args = parse_arguments()
    config = get_config(args)

    torch.set_float32_matmul_precision('medium')

    print("Intializing wandb run...")
    with wandb.init(
            project="agri-strat",
            job_type=args.job_type,
            notes=args.notes,
            tags=args.tags,
            resume="auto",
            config=config,
    ) as run:
        print("Creating datamodule, model, trainer...")
        datamodule = create_datamodule(config)
        model = create_model(config, datamodule)

        callbacks = [
            LearningRateMonitor(),
            ModelCheckpoint(
                dirpath=Path(wandb.run.dir) / "checkpoints",
                filename='ckpt_epoch={epoch:02d}',
                monitor=config["monitor_metric"],
                save_last='link',
                save_top_k=-1,
                mode='min',
                auto_insert_metric_name=False,
            ),
            CustomProgressBar(),
        ]
        trainer = pl.Trainer(
            accelerator="auto",
            devices=config["num_gpus"],
            num_nodes=config["num_nodes"],
            max_epochs=config["num_epochs"],
            check_val_every_n_epoch=1,
            precision='32-true',
            callbacks=callbacks,
            logger=WandbLogger(experiment=run),
            gradient_clip_val=10.0,
            deterministic=config["deterministic"],
            fast_dev_run=config["devtest"],
            limit_train_batches=config["limit_train_batches"],
            limit_val_batches=config["limit_val_batches"],
            num_sanity_val_steps=2,
            # profiler='simple',
        )

        print("Training...")
        trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
