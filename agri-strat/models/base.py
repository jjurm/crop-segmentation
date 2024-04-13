from pathlib import Path
from typing import Dict, Any, cast

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LRScheduler
from torch.utils.data.datapipes.iter.utils import IterableWrapperIterDataPipe
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix, \
    MulticlassPrecision, MulticlassRecall

from models.convstar import ConvSTAR
from models.unet import UNet
from utils.active_sampling.active_sampling import ActiveSampler
from utils.active_sampling.relevance_score.loss_score_fn import LossScoreFn
from utils.class_weights import ClassWeights
from utils.constants import IMG_SIZE
from utils.label_encoder import LabelEncoder
from utils.medians_metadata import MediansMetadata

NUM_VALIDATION_PATCH_EXAMPLES = 6
CLASS_LABEL_IGNORED = 255


def get_model_class(model):
    if model == "unet":
        model_class = UNet
    elif model == "convstar":
        model_class = ConvSTAR
    else:
        raise ValueError(f"model = {model}, expected: 'unet' or 'convstar'")
    return model_class


class BaseModelModule(pl.LightningModule):
    def __init__(
            self,
            model: str,
            label_encoder: LabelEncoder,
            learning_rate,
            class_weights: ClassWeights,
            bands: list[str],
            medians_metadata: MediansMetadata,
            batch_size: int,
            accumulate_grad_batches: int,
            n_batches_per_block: int,
            gradient_clip_val: float | None,
            parcel_loss=False,
            wandb_watch_log: str = None,
            **kwargs,
    ):
        """
        Parameters:
        -----------
        learning_rate: float, default 1e-3
            The initial learning rate.
            Use a weighted loss function with precalculated weights per class.
        parcel_loss: boolean, default False
            If True, then a custom loss function is used which takes into account
            only the pixels of the parcels. If False, then all image pixels are
            used in the loss function.
        """
        super(BaseModelModule, self).__init__()
        self.save_hyperparameters("model", "bands", "parcel_loss")

        self.label_encoder = label_encoder
        num_classes = label_encoder.num_classes
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.bands = bands
        self.batch_size = batch_size
        self.parcel_loss = parcel_loss
        self.medians_metadata = medians_metadata
        self.wandb_watch_log = wandb_watch_log
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val
        self.run_dir = Path(wandb.run.dir)

        self.monitor_metric = 'val/f1w_parcel' if self.parcel_loss else 'val/f1w'

        # Disable automatic optimization
        self.automatic_optimization = False

        # Log a table of class weights to Wandb
        wandb.log({"class_weights": self.class_weights.get_wandb_table()})

        # Global metrics
        self.loss_nll = None
        self.metric_acc = None
        self.metric_f1w = None
        self.metric_f1ma = None
        self.loss_nll = nn.NLLLoss(weight=self.class_weights.class_weights_weighted, reduction="none")
        if not self.parcel_loss:
            self.metric_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
            self.metric_f1w = MulticlassF1Score(num_classes=num_classes, average="weighted")
            self.metric_f1ma = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.metric_acc_parcel = MulticlassAccuracy(num_classes=num_classes, average="macro", ignore_index=0)
        self.metric_f1w_parcel = MulticlassF1Score(num_classes=num_classes, average="weighted", ignore_index=0)
        self.metric_f1ma_parcel = MulticlassF1Score(num_classes=num_classes, average="macro", ignore_index=0)
        self.metric_crop5_acc_parcel = MulticlassAccuracy(num_classes=num_classes, average="macro", ignore_index=0)
        self.metric_crop5_f1w_parcel = MulticlassF1Score(num_classes=num_classes, average="weighted", ignore_index=0)
        self.metric_crop5_f1ma_parcel = MulticlassF1Score(num_classes=num_classes, average="macro", ignore_index=0)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize="none",
                                                          ignore_index=0 if parcel_loss else None)
        # Per-class metrics
        self.metric_class_precision = MulticlassPrecision(average=None, num_classes=num_classes,
                                                          ignore_index=0 if parcel_loss else None)
        self.metric_class_recall = MulticlassRecall(average=None, num_classes=num_classes,
                                                    ignore_index=0 if parcel_loss else None)
        self.metric_class_f1 = MulticlassF1Score(average=None, num_classes=num_classes,
                                                 ignore_index=0 if parcel_loss else None)

        self.active_sampler = ActiveSampler(
            batch_size=batch_size,
            n_batches_per_block=n_batches_per_block,
            accumulate_grad_batches=accumulate_grad_batches,
            relevancy_score_fn=LossScoreFn(
                loss_fn=F.nll_loss,
                ignore_index=0 if parcel_loss else None,
                weight=self.class_weights.class_weights_weighted,
            ),
        )

        # Create the model
        self.model = get_model_class(model)(
            num_classes=num_classes,
            num_bands=len(bands),
            relative_class_frequencies=self.class_weights.relative_class_frequencies,
            **kwargs)

        self.num_samples_seen = 0
        self.num_pixels_seen = 0
        self.validation_examples = None
        self.validation_patch_scores = None
        self.train_samples_df = None

    @property
    def val_epoch(self):
        return self.current_epoch // self.trainer.check_val_every_n_epoch

    def setup(self, stage: str) -> None:
        wandb.run.summary["monitor_metric"] = self.monitor_metric

        step_metric = None
        if stage == "fit":
            step_metric = "trainer/global_step"
            wandb.define_metric(step_metric)
            wandb.define_metric("*", step_metric=step_metric)

        if stage == "fit" or stage == "validate":
            metrics_with_summaries = [
                "val/acc",
                "val/acc_parcel",
                "val/f1w",
                "val/f1w_parcel",
                "val/f1ma",
                "val/f1ma_parcel",
                "val/crop5_acc_parcel",
                "val/crop5_f1w_parcel",
                "val/crop5_f1ma_parcel",
            ]
            for metric in metrics_with_summaries:
                wandb.define_metric(metric, summary="max,last", step_metric=step_metric)

        # Log gradients
        if stage == "fit":
            wandb.watch(self.model, log_freq=100, log=self.wandb_watch_log)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        lr_scheduler: LRScheduler
        if self.hparams.get("model") == "unet":
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                threshold=1e-3,
                threshold_mode="rel",
                factor=0.5,
                patience=4,
            )
        elif self.hparams.get("model") == "convstar":
            lr_scheduler = StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            raise ValueError(f"model = {self.hparams.get('model')}, expected: 'unet' or 'convstar'")

        return [optimizer], [{
            'scheduler': lr_scheduler,
            'monitor': self.monitor_metric,
            'frequency': self.trainer.check_val_every_n_epoch,
        }]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["samples_seen"] = self.num_samples_seen
        checkpoint["pixels_seen"] = self.num_pixels_seen

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.num_samples_seen = n if (n := checkpoint["samples_seen"]) is not None else 0
        self.num_pixels_seen = n if (n := checkpoint["pixels_seen"]) is not None else 0

    def on_train_epoch_start(self) -> None:
        # TODO do not report train samples (only for debugging)
        self.train_samples_df = {
            "patch": [],
            "subpatch_x": [],
            "subpatch_y": [],
        }

    def training_step(self, block, block_idx):
        """
        :param block: [{
            'medians': torch.Tensor(shape=(T, C, H, W)),
            'labels': torch.Tensor(shape=(H, W)),
            ...,
        }] (len=B)
        :param block_idx: int
        """

        block_filtered = self.active_sampler(block, block_idx, self.model)
        batches = IterableWrapperIterDataPipe(block_filtered, deepcopy=False) \
            .batch(batch_size=self.batch_size) \
            .batch(batch_size=self.accumulate_grad_batches) \
            .collate()

        opt = cast(Optimizer, self.optimizers())
        for effective_batch in batches:
            opt.zero_grad()
            total_loss = 0.
            batch_size = 0

            # accumulate gradients of N batches
            for mini_batch in effective_batch:
                self._update_trainer_scalars(mini_batch)

                output = self.model(mini_batch['medians'])
                loss = self._compute_loss(output, mini_batch['labels'])

                total_loss += loss.item()
                batch_size += mini_batch['labels'].shape[0]

                # scale losses by 1/N regardless of the len(effective_batch) to keep the weight of samples constant
                loss = loss / self.accumulate_grad_batches
                self.manual_backward(loss)

            # clip gradients & update weights
            if self.gradient_clip_val is not None:
                self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
            opt.step()

            self._log_trainer_scalars()
            self.log('train/loss_nll_parcel', total_loss,
                     on_step=True, on_epoch=True, logger=True, prog_bar=self.parcel_loss, batch_size=batch_size)

    def _update_trainer_scalars(self, batch):
        labels = batch["labels"]
        batch_size = labels.shape[0]
        self.train_samples_df["patch"].extend(batch["patch_path"])
        self.train_samples_df["subpatch_x"].extend(batch["subpatch_yx"][1].cpu().numpy())
        self.train_samples_df["subpatch_y"].extend(batch["subpatch_yx"][0].cpu().numpy())

        self.num_samples_seen += batch_size
        if self.parcel_loss:
            self.num_pixels_seen += (labels != 0).sum().item()
        else:
            self.num_pixels_seen += batch_size * labels.shape[1] * labels.shape[2]

    def _log_trainer_scalars(self):
        wandb.log({
            'val_epoch': self.val_epoch,
            'trainer/samples_seen': self.num_samples_seen,
            'trainer/pixels_seen': self.num_pixels_seen,
        })

    def _compute_loss(self, output, labels):
        loss_nll_unreduced = self.loss_nll(output, labels)
        if not self.parcel_loss:
            raise NotImplementedError("parcel_loss=False is not implemented")
        # TODO change to .sum() then divide by the number of pixels to assign constant weight to each pixel.
        #  but probably log .mean()
        loss_nll_parcel = loss_nll_unreduced[labels != 0].mean()

        if torch.isnan(loss_nll_parcel):
            return None
        return loss_nll_parcel

    def on_train_epoch_end(self) -> None:
        self._export_train_samples_csv()
        self.train_samples_df = None

    def _export_train_samples_csv(self):
        csv_file = self.run_dir / "train_samples.csv"
        df = pd.DataFrame(self.train_samples_df)
        df["epoch"] = self.current_epoch
        df.to_csv(csv_file, index=False, mode='a', header=not csv_file.exists())

    def on_validation_epoch_start(self) -> None:
        self.validation_examples = {
            "patch": [],
            "inputs": [],
            "labels": [],
            "outputs": [],
        }
        self.validation_patch_scores = {}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['medians'], batch['labels']  # (B, T, C, H, W), (B, H, W)
        batch_size = inputs.shape[0]
        output = self.model(inputs)

        if self.trainer.state.stage != "sanity_check":
            wandb.log({
                'val_epoch': self.val_epoch,
                'trainer/samples_seen': self.num_samples_seen,
                'trainer/pixels_seen': self.num_pixels_seen,
            })

        # Loss
        loss_nll_unreduced = self.loss_nll(output, labels)

        loss_nll_parcel = loss_nll_unreduced[labels != 0].mean()
        if not torch.isnan(loss_nll_parcel):
            self.log('val/loss_nll_parcel', loss_nll_parcel,
                     on_step=False, on_epoch=True, logger=True, batch_size=batch_size)
        if not self.parcel_loss:
            loss_nll = loss_nll_unreduced.mean()
            if not torch.isnan(loss_nll):
                self.log('val/loss_nll', loss_nll,
                         on_step=False, on_epoch=True, logger=True, batch_size=batch_size)

        # Global metrics
        acc, f1w, f1ma = None, None, None
        if not self.parcel_loss:
            acc = self.metric_acc(output, labels)
            f1w = self.metric_f1w(output, labels)
            f1ma = self.metric_f1ma(output, labels)
        acc_parcel = self.metric_acc_parcel(output, labels)
        f1w_parcel = self.metric_f1w_parcel(output, labels)
        f1ma_parcel = self.metric_f1ma_parcel(output, labels)
        self.confusion_matrix.update(output, labels)

        # Crop5 metrics
        img_size_h, img_size_w = self.medians_metadata.img_size
        output_cropped = output[:, :, 5:img_size_h - 5, 5:img_size_w - 5]
        labels_cropped = labels[:, 5:img_size_h - 5, 5:img_size_w - 5]
        crop5_acc_parcel = self.metric_crop5_acc_parcel(output_cropped, labels_cropped)
        crop5_f1w_parcel = self.metric_crop5_f1w_parcel(output_cropped, labels_cropped)
        crop5_f1ma_parcel = self.metric_crop5_f1ma_parcel(output_cropped, labels_cropped)

        # Per-class metrics
        self.metric_class_precision.update(output, labels)
        self.metric_class_recall.update(output, labels)
        self.metric_class_f1.update(output, labels)

        # Per-patch metrics
        self._collect_per_patch_scores(batch, output)
        self._collect_preview_samples(batch, output)

        if not self.parcel_loss:
            self.log_dict({
                'val/acc': acc,
                'val/f1w': f1w,
                'val/f1ma': f1ma,
            }, on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=batch_size)
        self.log_dict({
            'val/acc_parcel': acc_parcel,
            'val/f1w_parcel': f1w_parcel,
            'val/f1ma_parcel': f1ma_parcel,
        }, on_step=False, on_epoch=True, logger=True, prog_bar=self.parcel_loss, batch_size=batch_size)
        self.log_dict({
            'val/crop5_acc_parcel': crop5_acc_parcel,
            'val/crop5_f1w_parcel': crop5_f1w_parcel,
            'val/crop5_f1ma_parcel': crop5_f1ma_parcel,
        }, on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=batch_size)

    def _collect_per_patch_scores(self, batch, output):
        for i, patch_path in enumerate(batch["patch_path"]):
            n_pixels = int(
                (batch["labels"][i] != 0).sum().item() if self.parcel_loss else batch["labels"][i].shape[0] *
                                                                                batch["labels"][i].shape[1])
            if n_pixels == 0:
                continue  # This is to avoid ending up with patches with no pixels of interest

            if patch_path not in self.validation_patch_scores:
                self.validation_patch_scores[patch_path] = {
                    "acc_parcel": MulticlassAccuracy(num_classes=self.label_encoder.num_classes,
                                                     average="macro", ignore_index=0).to(self.device),
                    "f1w_parcel": MulticlassF1Score(num_classes=self.label_encoder.num_classes,
                                                    average="weighted", ignore_index=0).to(self.device),
                    "f1ma_parcel": MulticlassF1Score(num_classes=self.label_encoder.num_classes,
                                                     average="macro", ignore_index=0).to(self.device),
                    "crop5_acc_parcel": MulticlassAccuracy(num_classes=self.label_encoder.num_classes,
                                                           average="macro", ignore_index=0).to(self.device),
                    "crop5_f1w_parcel": MulticlassF1Score(num_classes=self.label_encoder.num_classes,
                                                          average="weighted", ignore_index=0).to(self.device),
                    "crop5_f1ma_parcel": MulticlassF1Score(num_classes=self.label_encoder.num_classes,
                                                           average="macro", ignore_index=0).to(self.device),
                    "n_pixels": 0,
                }
                if not self.parcel_loss:
                    self.validation_patch_scores[patch_path] |= {
                        "acc": MulticlassAccuracy(num_classes=self.label_encoder.num_classes,
                                                  average="macro").to(self.device),
                        "f1w": MulticlassF1Score(num_classes=self.label_encoder.num_classes,
                                                 average="weighted").to(self.device),
                        "f1ma": MulticlassF1Score(num_classes=self.label_encoder.num_classes,
                                                  average="macro").to(self.device),
                    }
            scores = self.validation_patch_scores[patch_path]

            outputs_ = output[i].unsqueeze(0)
            labels_ = batch["labels"][i].unsqueeze(0)
            if not self.parcel_loss:
                scores["acc"].update(outputs_, labels_)
                scores["f1w"].update(outputs_, labels_)
                scores["f1ma"].update(outputs_, labels_)
            scores["acc_parcel"].update(outputs_, labels_)
            scores["f1w_parcel"].update(outputs_, labels_)
            scores["f1ma_parcel"].update(outputs_, labels_)
            img_size_h, img_size_w = self.medians_metadata.img_size
            outputs_cropped = outputs_[:, :, 5:img_size_h - 5, 5:img_size_w - 5]
            labels_cropped = labels_[:, 5:img_size_h - 5, 5:img_size_w - 5]
            scores["crop5_acc_parcel"].update(outputs_cropped, labels_cropped)
            scores["crop5_f1w_parcel"].update(outputs_cropped, labels_cropped)
            scores["crop5_f1ma_parcel"].update(outputs_cropped, labels_cropped)
            scores["n_pixels"] += n_pixels

    def _collect_preview_samples(self, batch, output):
        # Collect validation examples to show as images
        want_examples = NUM_VALIDATION_PATCH_EXAMPLES * self.medians_metadata.num_subpatches_per_patch
        has_examples = len(self.validation_examples["inputs"])
        if has_examples < want_examples:
            self.validation_examples["patch"].extend(batch["patch_path"][:(want_examples - has_examples)])
            self.validation_examples["inputs"].extend(
                batch["medians"][:(want_examples - has_examples)].cpu().detach().numpy())
            self.validation_examples["labels"].extend(
                batch["labels"][:(want_examples - has_examples)].cpu().detach().numpy())
            self.validation_examples["outputs"].extend(
                output[:(want_examples - has_examples)].cpu().detach().numpy())

    def on_validation_epoch_end(self) -> None:
        confusion_matrix_cpu = self.confusion_matrix.compute().cpu()
        wandb_cm = self._get_wandb_confusion_matrix(confusion_matrix_cpu)
        class_scores_df = self._compute_class_scores_table(confusion_matrix_cpu)
        patch_scores_df = self._compute_per_patch_scores()

        if self.trainer.state.stage != "sanity_check":
            examples_table, images = self._get_preview_table_and_samples()
            wandb.log({
                "epoch": self.current_epoch,
                "val_epoch": self.val_epoch,
                "confusion_matrix": wandb_cm,
                "examples": images,
                "examples_table": examples_table,
                "class_scores": wandb.Table(dataframe=class_scores_df),
                "patch_scores": wandb.Table(dataframe=patch_scores_df),
            })

        self.validation_examples = None
        self.validation_patch_scores = None

    def _get_wandb_confusion_matrix(self, confusion_matrix_cpu):
        data = []
        classes = range(self.confusion_matrix.num_classes)[int(self.parcel_loss):]
        for i in classes:
            for j in classes:
                data.append([
                    self.label_encoder.class_names[i],
                    self.label_encoder.class_names[j],
                    confusion_matrix_cpu[i, j].item(),
                ])

        fields = {
            "Actual": "Actual",
            "Predicted": "Predicted",
            "nPredictions": "nPredictions",
        }
        return wandb.plot_table(
            "wandb/confusion_matrix/v1",
            wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=data),
            fields,
            {"title": f"Confusion Matrix"},
        )

    def _compute_class_scores_table(self, confusion_matrix_cpu):
        start_i = int(self.parcel_loss)
        class_counts = confusion_matrix_cpu.sum(axis=1).numpy()
        class_precision = self.metric_class_precision.compute().cpu().numpy()
        class_recall = self.metric_class_recall.compute().cpu().numpy()
        class_f1 = self.metric_class_f1.compute().cpu().numpy()
        per_class_scores_df = pd.DataFrame({
            "class": self.label_encoder.class_names[start_i:],
            "pixel_count": class_counts[start_i:],
            "precision": class_precision[start_i:],
            "recall": class_recall[start_i:],
            "f1": class_f1[start_i:],
        }, index=range(start_i, self.label_encoder.num_classes))
        return per_class_scores_df

    def _compute_per_patch_scores(self):
        for patch_path, scores in self.validation_patch_scores.items():
            if not self.parcel_loss:
                scores["acc"] = scores["acc"].compute().item()
                scores["f1w"] = scores["f1w"].compute().item()
                scores["f1ma"] = scores["f1ma"].compute().item()
            scores["acc_parcel"] = scores["acc_parcel"].compute().item()
            scores["f1w_parcel"] = scores["f1w_parcel"].compute().item()
            scores["f1ma_parcel"] = scores["f1ma_parcel"].compute().item()
            scores["crop5_acc_parcel"] = scores["crop5_acc_parcel"].compute().item()
            scores["crop5_f1w_parcel"] = scores["crop5_f1w_parcel"].compute().item()
            scores["crop5_f1ma_parcel"] = scores["crop5_f1ma_parcel"].compute().item()
        patch_scores_df = pd.DataFrame.from_dict(self.validation_patch_scores, orient="index").reset_index()
        csv_file = self.run_dir / "patch_scores.csv"
        patch_scores_df.to_csv(csv_file, index=False)
        return patch_scores_df

    def _get_preview_table_and_samples(self):
        # This should hold for all validation epochs with enough of examples
        # and only fail for cases like sanity check or a devtest run
        needed_example = NUM_VALIDATION_PATCH_EXAMPLES * self.medians_metadata.num_subpatches_per_patch
        assert len(self.validation_examples["patch"]) == needed_example, (
            f"Not enough subpatches for validation examples: got {len(self.validation_examples['patch'])} < "
            f"{needed_example} wanted"
        )

        examples_table = wandb.Table(columns=["patch", "inputs", "ground truth", "predictions", "errors"])
        rgb_band_indices = [self.bands.index(band) for band in ["B04", "B03", "B02"]]
        images = []
        class_labels = {i: name for i, _, name in self.label_encoder.entries_with_name}
        error_labels = {
            0: "correct",
            1: "incorrect",
        }
        if self.parcel_loss:
            class_labels[CLASS_LABEL_IGNORED] = "ignored"
            error_labels[CLASS_LABEL_IGNORED] = "ignored"
        class_set = [
            {"id": i, "name": name}
            for i, name in class_labels.items()
        ]

        pixels_dtype = self.validation_examples["inputs"][0].dtype
        labels_dtype = self.validation_examples["labels"][0].dtype
        predictions_dtype = self.validation_examples["outputs"][0].dtype
        for i_patch in range(NUM_VALIDATION_PATCH_EXAMPLES):
            # The following assumes that each patch has its 'num_subpatches_per_patch' subpatches in a sequence
            # i.e. that the subpatches are not shuffled within a patch
            patch = self.validation_examples["patch"][i_patch * self.medians_metadata.num_subpatches_per_patch]
            patch_size = (self.medians_metadata.patch_size[0], self.medians_metadata.patch_size[1]) \
                if self.medians_metadata.patch_size else (IMG_SIZE, IMG_SIZE)
            pixels = np.zeros(patch_size + (3,), dtype=pixels_dtype)  # (H, W, C)
            labels = np.zeros(patch_size, dtype=labels_dtype)  # (H, W)
            predictions = np.zeros(patch_size, dtype=predictions_dtype)  # (H, W)

            # Put together the image of the whole patch
            y_rows = patch_size[1] // self.medians_metadata.img_size[1]
            for i_subpatch in range(self.medians_metadata.num_subpatches_per_patch):
                i_subpatch_in_examples = i_patch * self.medians_metadata.num_subpatches_per_patch + i_subpatch
                # Shapes of variables:
                #   inputs (T, C, H, W)
                #   labels (H, W)
                #   outputs (P, H, W)
                inputs_subpatch = self.validation_examples["inputs"][i_subpatch_in_examples]
                pixels_subpatch = np.stack([
                    inputs_subpatch[3, band_index, :, :]  # 3 is the time index
                    for band_index in rgb_band_indices
                ], axis=-1)  # Channel should be last, i.e. (H, W, C)

                low_x = (i_subpatch // y_rows) * self.medians_metadata.img_size[0]
                high_x = low_x + self.medians_metadata.img_size[0]
                low_y = (i_subpatch % y_rows) * self.medians_metadata.img_size[1]
                high_y = low_y + self.medians_metadata.img_size[1]
                pixels[low_x:high_x, low_y:high_y, :] = pixels_subpatch
                labels[low_x:high_x, low_y:high_y] = self.validation_examples["labels"][i_subpatch_in_examples]
                predictions[low_x:high_x, low_y:high_y] = self.validation_examples["outputs"][i_subpatch_in_examples] \
                    .argmax(axis=0)

            # Normalize the range of each image to [0,1]
            pixels = pixels / np.max(pixels)
            pixels_scaled = np.floor(pixels * 256).clip(0, 255)
            patch_name = Path(patch).stem

            # noinspection PyUnresolvedReferences
            error_pixels = (labels != predictions).astype(int)
            if self.parcel_loss:
                predictions[labels == 0] = CLASS_LABEL_IGNORED  # 'ignored' class
                error_pixels[labels == 0] = CLASS_LABEL_IGNORED  # 'ignored' class
            ground_truth_mask = {"ground_truth": dict(mask_data=labels, class_labels=class_labels)}
            prediction_mask = {"prediction": dict(mask_data=predictions, class_labels=class_labels)}
            error_mask = {"error": dict(mask_data=error_pixels, class_labels=error_labels)}
            images.append(wandb.Image(
                pixels_scaled,
                caption=f"({i_patch + 1}) {patch_name}",
                masks=ground_truth_mask | prediction_mask | error_mask,
                classes=class_set,
            ))

            examples_table.add_data(
                patch_name,
                wandb.Image(pixels_scaled, caption=f"inputs ({i_patch + 1})"),
                wandb.Image(pixels_scaled, caption=f"ground truth ({i_patch + 1})",
                            masks=ground_truth_mask),
                wandb.Image(pixels_scaled, caption=f"prediction ({i_patch + 1})",
                            masks=prediction_mask),
                wandb.Image(pixels_scaled, caption=f"errors ({i_patch + 1})",
                            masks=error_mask),
            )

        return examples_table, images
