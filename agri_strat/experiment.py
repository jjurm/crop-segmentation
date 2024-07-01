#!/usr/bin/env python

__author__ = "Juraj Micko"
__license__ = "MIT License"

if __name__ == '__main__':
    print("Importing modules...")

import argparse
import lightning.pytorch as pl
import os
import torch
import wandb
from agri_strat.models.base import BaseModelModule
from agri_strat.utils.callbacks.batch_counter import BatchCounterCallback
from agri_strat.utils.callbacks.custom_lr_monitor import CustomLearningRateMonitor
from agri_strat.utils.callbacks.custom_model_checkpoint import CustomModelCheckpoint
from agri_strat.utils.callbacks.print_stage import PrintStageCallback
from agri_strat.utils.class_weights_module import ClassWeights
from agri_strat.utils.custom_progress_bar import CustomProgressBar
from agri_strat.utils.custom_wandb_logger import CustomWandbLogger
from agri_strat.utils.exception_tracker_callback import ExceptionTrackerCallback
from agri_strat.utils.label_encoder import LabelEncoder
from agri_strat.utils.medians_datamodule import MediansDataModule
from lightning import seed_everything
from math import ceil
from pathlib import Path


def parse_arguments():
    # Parse user arguments
    parser_with_groups = argparse.ArgumentParser()

    parser = parser_with_groups.add_argument_group('job')
    parser.add_argument('--job_type', type=str, default="train", required=False,
                        choices=['train', 'val', 'test', 'train_test'],
                        help='Type of job to perform. One of [\'train\', \'val\', \'test\']')
    parser.add_argument('--tags', nargs="+", default=None, required=False,
                        help='Tags stored with the wandb job.')
    parser.add_argument('--notes', type=str, default=None, required=False,
                        help='Note stored with the wandb job.')

    parser = parser_with_groups.add_argument_group('data')
    parser.add_argument('--label_encoder_artifact', type=str, default="s4a_labels:latest", required=False,
                        help='Wandb artifact of type \'label_encoder\' that defines classes to be predicted.')
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
    parser.add_argument('--requires_norm', action='store_true', default=False, required=False,
                        help='Normalize data to 0-1 range. Default False')

    parser = parser_with_groups.add_argument_group('data_loading')
    parser.add_argument('--seed', type=int, default=0, required=False, )
    parser.add_argument('--skip_empty_subpatches', action='store_true', default=False, required=False,
                        help='Skip subpatches during training that have no pixels of interest (only relevant with '
                             'parcel_loss). Default False.')
    parser.add_argument('--shuffle_subpatches_within_patch', action='store_true', default=False, required=False,
                        help='Shuffle subpatches within each patch (only applies to training). Default False.')
    parser.add_argument('--num_workers', type=int, default=6, required=False,
                        help='Number of workers to work on dataloader. Default 6')
    parser.add_argument('--shuffle_buffer_num_patches', type=int, default=0, required=False,
                        help='Size of the buffer for shuffling subpatches, given in the number of patches. Default 0 '
                             '(no shuffling).')

    parser = parser_with_groups.add_argument_group('model')
    parser.add_argument('--model', type=str, default="unet", required=False,
                        choices=['unet', 'convstar'],
                        help='Model to use. One of [\'unet\', \'convstar\']')
    parser.add_argument("--num_layers", type=int, default=3, required=False,
                        help="Number of layers in the model. Default 3.")
    parser.add_argument("--load_model", type=str, default=None, required=False,
                        help="Name of the artifact to load the model from, used for job_type: val or test. Default "
                             "None.")

    parser = parser_with_groups.add_argument_group('loss')
    parser.add_argument('--parcel_loss', action='store_true', default=False, required=False,
                        help='Use a loss function that takes into account parcel pixels only.')
    parser.add_argument('--weighted_loss', action='store_true', default=False, required=False,
                        help='Use a weighted loss function with precalculated weights per class. Default False.')
    parser.add_argument('--class_weights_weight', type=float, default=1.0, required=False,
                        help='Weight of the class weights in the loss function, interpolating between calculated '
                             'class weights and uniform weights. Default 1.0 = use calculated class weights.')

    parser = parser_with_groups.add_argument_group('training')
    parser.add_argument('--num_epochs', type=int, default=10, required=False,
                        help='Number of epochs. Also scaled by --block_size. Default 10')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, required=False,
                        help='Check validation every n epochs. Also scaled by --block_size. Default 1')
    parser.add_argument('--batch_size', type=int, default=4, required=False,
                        help='If physical_batch_size=None, behaves as the conventional batch_size. Otherwise, '
                             'sets the effective batch size and must be smaller than or a multiply of '
                             'physical_batch_size. Default 4')
    parser.add_argument('--physical_batch_size', type=int, default=None, required=False,
                        help='If set and smaller than batch_size, sets the batch_size the model receives in each step.')
    parser.add_argument('--learning_rate', type=float, default=1e-1, required=False,
                        help='Starting learning rate. Default 1e-1')
    parser.add_argument('--gradient_clip_val', type=float, default=10.0, required=False,
                        help='Gradient clipping value. Default 10.0')
    parser.add_argument('--deterministic', action='store_true', default=False, required=False,
                        help='Enforce reproducible results (except functions without a deterministic implementation). '
                             'Default False')

    parser = parser_with_groups.add_argument_group('hardware')
    parser.add_argument('--num_gpus', type=int, default=1, required=False,
                        help='Number of gpus to use (per node). Default 1')
    parser.add_argument('--num_nodes', type=int, default=1, required=False,
                        help='Number of nodes to use. Default 1')

    parser = parser_with_groups.add_argument_group('development')
    parser.add_argument('--devtest', action='store_true', default=False, required=False,
                        help='Perform a dev test run with this model')
    parser.add_argument('--limit_train_batches', type=float, default=None, required=False,
                        help='Limit the number of batches to run during training (float = fraction, '
                             'int = num_batches). Default None')
    parser.add_argument('--limit_val_batches', type=float, default=None, required=False,
                        help='Limit the number of batches to run during validation (float = fraction, '
                             'int = num_batches). Default None')

    parser = parser_with_groups.add_argument_group('active_sampling')
    parser.add_argument('--block_size', type=float, default=1.0, required=False,
                        help='The size of an active sampling block, given as a multiplier of the number of samples to '
                             'be fed to the model; can be fractional. block_size=2.0 means that the number of samples '
                             'considered will be double the number of samples fed to the model. This also scales the '
                             '--num_epochs and --check_val_every_n_epoch parameters proportionally. When set to 1.0 '
                             '(default), the model will not be trained with active sampling. Default 1.0')
    parser.add_argument('--n_batches_per_block', type=int, default=1, required=False,
                        help='The number of batches (of effective batch_size) to sample in each active sampling '
                             'block. Controls the active sampling frequency. Default 1, i.e. sample every batch')
    parser.add_argument('--active_sampling_relevancy_score', type=str, nargs="+", default=["none"],
                        help='The relevancy score function to use for active sampling. Choices: ["none", "loss", '
                             '"rho-loss-<irreducible_loss_model_artifact>", "uncertainty-margin", "random"], or a list '
                             'of "<float_weight>*<fn>". Default none')

    parser = parser_with_groups.add_argument_group('logging')
    parser.add_argument('--wandb_watch_log', type=str, default=None, required=False,
                        choices=["gradients", "parameters", "all"],
                        help='Log gradients, parameters or both. Default None')
    parser.add_argument('--eval_every_n_val_epoch', type=int, default=0, required=False,
                        help='Evaluate the model with more stats every n val_epochs (per-class scores, per-patch '
                             'scores, preview samples). Always evaluates at the end. Set to 0 to only compute these '
                             'stats in the last val_epoch. Default 0')

    return parser_with_groups.parse_args()


def get_config(args):
    exclude_keys = {"tags", "notes", "job_type"}
    config = {k: v for k, v in vars(args).items() if k not in exclude_keys}
    return config


def create_datamodule(config, label_encoder, calculated_batch_size, accumulate_grad_batches):
    datamodule = MediansDataModule(
        medians_artifact=config["medians_artifact"],
        medians_path=config["medians_path"] or os.getenv("MEDIANS_PATH", "dataset/medians"),
        split_artifact=config["split_artifact"],
        bins_range=config["bins_range"],
        label_encoder=label_encoder,
        requires_norm=config["requires_norm"],
        batch_size=calculated_batch_size,  # val and test sets don't use blocks
        block_size=round(config["block_size"] * config["n_batches_per_block"] * config["batch_size"]),
        num_workers=config["num_workers"],
        seed=config["seed"],
        shuffle_buffer_num_patches=config["shuffle_buffer_num_patches"],
        skip_zero_label_subpatches=config["parcel_loss"] and config["skip_empty_subpatches"],
        # integer limits are passed directly to the trainer instead
        limit_train_batches=limit if (limit := config["limit_train_batches"]) and limit % 1.0 != 0 else None,
        limit_val_batches=limit if (limit := config["limit_val_batches"]) and limit % 1.0 != 0 else None,
        shuffle_subpatches_within_patch=config["shuffle_subpatches_within_patch"],
    )
    datamodule.prepare_data()
    return datamodule


def get_ckpt_path(run, checkpoint_name):
    ckpt_path = None
    if run.resumed:
        if "epoch" in run.summary.keys():
            ckpt_artifact = wandb.run.use_artifact("{checkpoint_name}:epoch={epoch:02d}".format(
                checkpoint_name=checkpoint_name,
                epoch=run.summary["epoch"],
            ))
            ckpt_path = ckpt_artifact.file()
            print(f"Resuming run {run.name}, loaded a checkpoint from epoch={run.summary['epoch']}.")
        else:
            print(f"Resumed run {run.name} is empty, starting all from scratch.")
    if ckpt_path is None:
        if run.config["load_model"]:
            ckpt_artifact = wandb.run.use_artifact(run.config["load_model"], type="model")
            ckpt_path = ckpt_artifact.file()
            print(f"Loaded model from artifact {run.config['load_model']}.")
    return ckpt_path


def create_model(config, label_encoder: LabelEncoder, datamodule: MediansDataModule, **kwargs):
    a_s_relevancy_score = spec if isinstance((spec := config["active_sampling_relevancy_score"]), list) else [spec]
    unsaved_params = dict(
        label_encoder=label_encoder,
        bands=datamodule.get_bands(),
        num_time_steps=config["bins_range"][1] - config["bins_range"][0] + 1,
        medians_metadata=datamodule.metadata,
        wandb_watch_log=config["wandb_watch_log"],
        active_sampling_relevancy_score=a_s_relevancy_score,
        eval_every_n_val_epoch=config["eval_every_n_val_epoch"],
    ) | kwargs

    # Create a new model
    return BaseModelModule(
        model=config["model"],
        parcel_loss=config["parcel_loss"],
        num_layers=config["num_layers"],
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
        label_encoder = LabelEncoder(run.use_artifact(run.config["label_encoder_artifact"], type="label_encoder"))
        datamodule = create_datamodule(
            config=run.config,
            label_encoder=label_encoder,
            calculated_batch_size=calculated_batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
        )
        class_weights = ClassWeights(class_counts=datamodule.pixel_counts['train'], label_encoder=label_encoder,
                                     parcel_loss=run.config["parcel_loss"], weighted_loss=run.config["weighted_loss"],
                                     class_weights_weight=run.config["class_weights_weight"])

        checkpoint_name = f"model-{run.name}"
        ckpt_path = get_ckpt_path(run, checkpoint_name)
        model = create_model(
            config=run.config,
            label_encoder=label_encoder,
            datamodule=datamodule,
            class_weights=class_weights,
            batch_size=calculated_batch_size,  # train blocks need to be batched
            accumulate_grad_batches=accumulate_grad_batches,
            n_batches_per_block=run.config["n_batches_per_block"],
            gradient_clip_val=run.config["gradient_clip_val"],
        )

        callbacks = [
            PrintStageCallback(),
            BatchCounterCallback(
                datamodule,
                # The following scenario is not supported:
                # - each epoch has different order of patches
                # - skip_empty_subpatches is True
                # - limit_train_batches is set
                # In that case, the second epoch might have more batches than the first one,
                # resulting in no batch being is_last_batch. Problem is avoided by not setting the number of batches
                # in the trainer.
                set_trainer_max_batches=not (run.config["skip_empty_subpatches"] and run.config["limit_train_batches"]),
            ),
            ExceptionTrackerCallback(),
            CustomLearningRateMonitor(),
            model_checkpoint := CustomModelCheckpoint(
                dirpath=Path(wandb.run.dir) / "checkpoints",
                filename='ckpt_epoch={epoch:02d}',
                monitor=model.monitor_metric,
                save_last=False,
                save_top_k=-1,
                mode='max',
                auto_insert_metric_name=False,
            ),
            CustomProgressBar(refresh_rate=1 if run.config["devtest"] else 10),
        ]
        logger = CustomWandbLogger(
            experiment=run,
            log_model='all',
            checkpoint_name=checkpoint_name,
        )

        trainer = pl.Trainer(
            accelerator="auto",
            devices=run.config["num_gpus"],
            num_nodes=run.config["num_nodes"],
            max_epochs=ceil(run.config["num_epochs"] * run.config["block_size"]),
            check_val_every_n_epoch=round(run.config["check_val_every_n_epoch"] * run.config["block_size"]),
            precision='32-true',
            callbacks=callbacks,
            logger=logger,
            deterministic="warn" if run.config["deterministic"] else None,
            benchmark=not run.config["deterministic"],
            fast_dev_run=run.config["devtest"],
            # fractional batch limits are passed directly to the data module instead
            limit_train_batches=int(limit) if (limit := run.config[
                "limit_train_batches"]) and limit % 1.0 == 0 else None,
            limit_val_batches=int(limit) if (limit := run.config["limit_val_batches"]) and limit % 1.0 == 0 else None,
            num_sanity_val_steps=2,
            # profiler="simple",
            log_every_n_steps=20,
        )

        run_job(args.job_type, trainer, model, datamodule, ckpt_path, model_checkpoint)


def run_job(
        job_type: str,
        trainer: pl.Trainer,
        model: BaseModelModule,
        datamodule: MediansDataModule,
        ckpt_path: str | None,
        model_checkpoint: CustomModelCheckpoint,
):
    kwargs = dict(
        model=model,
        datamodule=datamodule,
    )

    if job_type == 'train':
        trainer.fit(ckpt_path=ckpt_path, **kwargs)

    elif job_type == 'train_test':
        trainer.fit(ckpt_path=ckpt_path, **kwargs)
        best_model_path = model_checkpoint.best_model_path
        trainer.test(ckpt_path=best_model_path, **kwargs)

    elif job_type == 'val':
        trainer.validate(ckpt_path=ckpt_path, **kwargs)

    elif job_type == 'test':
        trainer.validate(ckpt_path=ckpt_path, **kwargs)
        trainer.test(ckpt_path=ckpt_path, **kwargs)

    else:
        raise ValueError(f"Invalid job type: {job_type}")


if __name__ == '__main__':
    main()
