from pathlib import Path
from typing import Dict, Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LRScheduler
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix

from models.convstar import ConvSTAR
from models.unet import UNet
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
            weighted_loss: bool,
            class_weights_weight: float,  # interpolate between class weights and uniform weights
            bands: list[str],
            medians_metadata: MediansMetadata,
            parcel_loss=False,
            class_counts: dict[int, int] = None,
            **kwargs,
    ):
        """
        Parameters:
        -----------
        learning_rate: float, default 1e-3
            The initial learning rate.
        weighted_loss: boolean
            Use a weighted loss function with precalculated weights per class.
        parcel_loss: boolean, default False
            If True, then a custom loss function is used which takes into account
            only the pixels of the parcels. If False, then all image pixels are
            used in the loss function.
        class_counts: dict, default None
            Counts of pixels per class to use for class weights calculation for the loss function.
        """
        super(BaseModelModule, self).__init__()
        self.save_hyperparameters(
            ignore=["class_counts", "label_encoder", "bands", "num_time_steps", "medians_metadata"])

        self.label_encoder = label_encoder
        num_classes = label_encoder.num_classes
        self.learning_rate = learning_rate
        self.bands = bands
        self.parcel_loss = parcel_loss
        self.medians_metadata = medians_metadata

        self.monitor_metric = 'val/f1w_parcel' if self.parcel_loss else 'val/f1w'

        # Calculate class weights and log as a table to Wandb
        class_counts, relative_class_frequencies = self._calculate_class_counts_frequencies(class_counts, parcel_loss)
        class_weights = self._calculate_class_weights(relative_class_frequencies)
        self._log_class_counts_weights(class_counts, class_weights, parcel_loss)

        # Loss function
        if weighted_loss:
            class_weights_weighted = class_weights_weight * class_weights + \
                                     (1 - class_weights_weight) * torch.ones(num_classes)
            class_weights_weighted = class_weights_weighted.float().cuda()
        else:
            class_weights_weighted = None

        self.loss_nll = nn.NLLLoss(weight=class_weights_weighted)
        self.loss_nll_parcel = nn.NLLLoss(weight=class_weights_weighted, ignore_index=0)
        self.metric_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.metric_acc_parcel = MulticlassAccuracy(num_classes=num_classes, average="macro", ignore_index=0)
        self.metric_f1w = MulticlassF1Score(num_classes=num_classes, average="weighted")
        self.metric_f1w_parcel = MulticlassF1Score(num_classes=num_classes, average="weighted", ignore_index=0)
        self.metric_f1ma = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.metric_f1ma_parcel = MulticlassF1Score(num_classes=num_classes, average="macro", ignore_index=0)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize="none",
                                                          ignore_index=0 if parcel_loss else None)

        self.run_dir = Path(wandb.run.dir)

        # Create the model
        self.model = get_model_class(model)(
            num_classes=num_classes,
            num_bands=len(bands),
            relative_class_frequencies=relative_class_frequencies,
            **kwargs)

        self.num_pixels_seen = 0
        self.validation_examples = None
        self.validation_patch_scores = None

    def _calculate_class_weights(self, relative_class_frequencies):
        """
        Calculate class weights for the loss function.
        """
        n_classes = len(relative_class_frequencies)
        class_weights = [
            1 / (n_classes * freq) if freq != 0 else 0
            for freq in relative_class_frequencies
        ]

        return torch.tensor(class_weights)

    def _log_class_counts_weights(self, class_counts, class_weights, parcel_loss):
        class_weights_table = wandb.Table(
            columns=["IDs", "Class Name", "Pixel count", "Weight"],
            data=[
                [",".join(map(str, dataset_labels)), name, class_counts[i], class_weights[i]]
                for i, dataset_labels, name in self.label_encoder.entries_with_name
                if (i != 0 or not parcel_loss)  # if parcel_loss, skip the zero class
            ],
        )
        wandb.log({"class_weights": class_weights_table})

    def _calculate_class_counts_frequencies(self, class_counts: dict[int, int], parcel_loss):
        mapped_counts = np.array([
            0 if i == 0 and parcel_loss else
            sum(class_counts[dataset_label] for dataset_label in dataset_labels)
            for i, dataset_labels in self.label_encoder.entries
        ])
        total_count = np.sum(mapped_counts)
        relative_class_frequencies = mapped_counts / total_count
        return mapped_counts, relative_class_frequencies

    def setup(self, stage: str) -> None:
        wandb.run.summary["monitor_metric"] = self.monitor_metric

        # Define metric summaries
        if stage == "fit" or stage == "validate":
            wandb.define_metric("val/acc", summary="max,mean,last")
            wandb.define_metric("val/acc_parcel", summary="max,mean,last")
            wandb.define_metric("val/f1w", summary="max,mean,last")
            wandb.define_metric("val/f1w_parcel", summary="max,mean,last")
            wandb.define_metric("val/f1ma", summary="max,mean,last")
            wandb.define_metric("val/f1ma_parcel", summary="max,mean,last")

        # Log gradients
        if stage == "fit":
            wandb.watch(self.model, log_freq=100, log="gradients")

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
        }]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["pixels_seen"] = self.num_pixels_seen

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.num_pixels_seen = checkpoint["pixels_seen"]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch['medians'], batch['labels']  # (B, T, C, H, W), (B, H, W)
        batch_size = inputs.shape[0]
        output = self.model(inputs)
        loss_nll = self.loss_nll(output, labels)
        loss_nll_parcel = self.loss_nll_parcel(output, labels)
        self.log('train/loss_nll', loss_nll,
                 on_step=True, on_epoch=True, logger=True, prog_bar=not self.parcel_loss, batch_size=batch_size)
        self.log('train/loss_nll_parcel', loss_nll_parcel,
                 on_step=True, on_epoch=True, logger=True, prog_bar=self.parcel_loss, batch_size=batch_size)

        if self.parcel_loss:
            self.num_pixels_seen += (labels != 0).sum().item()
        else:
            self.num_pixels_seen += batch_size * labels.shape[1] * labels.shape[2]
        self.log_dict({
            'trainer/samples_seen': self.global_step * batch_size,
            'trainer/pixels_seen': self.num_pixels_seen,
        }, on_step=True, on_epoch=False, logger=True, batch_size=batch_size)

        loss = loss_nll_parcel if self.parcel_loss else loss_nll
        if torch.isnan(loss):
            return None
        return loss

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

        loss_nll = self.loss_nll(output, labels)
        loss_nll_parcel = self.loss_nll_parcel(output, labels)
        if not torch.isnan(loss_nll):
            self.log('val/loss_nll', loss_nll,
                     on_step=False, on_epoch=True, logger=True, batch_size=batch_size)
        if not torch.isnan(loss_nll_parcel):
            self.log('val/loss_nll_parcel', loss_nll_parcel,
                     on_step=False, on_epoch=True, logger=True, batch_size=batch_size)

        acc = self.metric_acc(output, labels)
        f1w = self.metric_f1w(output, labels)
        f1ma = self.metric_f1ma(output, labels)
        acc_parcel = self.metric_acc_parcel(output, labels)
        f1w_parcel = self.metric_f1w_parcel(output, labels)
        f1ma_parcel = self.metric_f1ma_parcel(output, labels)
        self.confusion_matrix.update(output, labels)

        self._collect_per_patch_scores(batch, output)
        self._collect_preview_samples(batch, output)

        self.log_dict({
            'val/acc': acc,
            'val/f1w': f1w,
        }, on_step=False, on_epoch=True, logger=True, prog_bar=not self.parcel_loss, batch_size=batch_size)
        self.log_dict({
            'val/acc_parcel': acc_parcel,
            'val/f1w_parcel': f1w_parcel,
        }, on_step=False, on_epoch=True, logger=True, prog_bar=self.parcel_loss, batch_size=batch_size)
        self.log_dict({
            'val/f1ma': f1ma,
            'val/f1ma_parcel': f1ma_parcel,
        }, on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=batch_size)

    def _collect_per_patch_scores(self, batch, output):
        for i, patch_path in enumerate(batch["patch_path"]):
            if patch_path not in self.validation_patch_scores:
                self.validation_patch_scores[patch_path] = {
                    "acc": MulticlassAccuracy(num_classes=self.label_encoder.num_classes,
                                              average="macro").to(self.device),
                    "acc_parcel": MulticlassAccuracy(num_classes=self.label_encoder.num_classes,
                                                     average="macro", ignore_index=0).to(self.device),
                    "f1w": MulticlassF1Score(num_classes=self.label_encoder.num_classes,
                                             average="weighted").to(self.device),
                    "f1w_parcel": MulticlassF1Score(num_classes=self.label_encoder.num_classes,
                                                    average="weighted", ignore_index=0).to(self.device),
                    "f1ma": MulticlassF1Score(num_classes=self.label_encoder.num_classes,
                                              average="macro").to(self.device),
                    "f1ma_parcel": MulticlassF1Score(num_classes=self.label_encoder.num_classes,
                                                     average="macro", ignore_index=0).to(self.device),
                    "n_pixels": 0,
                }
            scores = self.validation_patch_scores[patch_path]

            n_pixels = int(
                (batch["labels"][i] != 0).sum().item() if self.parcel_loss else batch["labels"][i].shape[0] *
                                                                                batch["labels"][i].shape[1])
            outputs_ = output[i].unsqueeze(0)
            labels_ = batch["labels"][i].unsqueeze(0)
            scores["acc"].update(outputs_, labels_)
            scores["f1w"].update(outputs_, labels_)
            scores["f1ma"].update(outputs_, labels_)
            scores["acc_parcel"].update(outputs_, labels_)
            scores["f1w_parcel"].update(outputs_, labels_)
            scores["f1ma_parcel"].update(outputs_, labels_)
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
        confusion_matrix = self.confusion_matrix.compute()
        wandb_cm = self._get_wandb_confusion_matrix(confusion_matrix)
        patch_scores_df = self._compute_per_patch_scores()

        if self.trainer.state.stage != "sanity_check":
            examples_table, images = self._get_preview_table_and_samples()
            wandb.log({
                "epoch": self.current_epoch,
                "confusion_matrix": wandb_cm,
                "examples": images,
                "examples_table": examples_table,
                "patch_scores": wandb.Table(dataframe=patch_scores_df),
            })

        self.validation_examples = None
        self.validation_patch_scores = None

    def _get_wandb_confusion_matrix(self, confusion_matrix):
        data = []
        classes = range(self.confusion_matrix.num_classes)[int(self.parcel_loss):]
        for i in classes:
            for j in classes:
                data.append([
                    self.label_encoder.class_names[i],
                    self.label_encoder.class_names[j],
                    confusion_matrix[i, j],
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

    def _compute_per_patch_scores(self):
        for patch_path, scores in self.validation_patch_scores.items():
            scores["acc"] = scores["acc"].compute().item()
            scores["f1w"] = scores["f1w"].compute().item()
            scores["f1ma"] = scores["f1ma"].compute().item()
            scores["acc_parcel"] = scores["acc_parcel"].compute().item()
            scores["f1w_parcel"] = scores["f1w_parcel"].compute().item()
            scores["f1ma_parcel"] = scores["f1ma_parcel"].compute().item()
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
