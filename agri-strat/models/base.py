import pickle
from pathlib import Path

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from matplotlib.patches import Rectangle
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LRScheduler
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix

from config import SELECTED_CLASSES
from models.convstar import ConvSTAR
from models.unet import UNet


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
            linear_encoder,
            learning_rate,
            monitor_metric: str,
            weighted_loss: bool,
            bands: list[str],
            parcel_loss=False,
            class_counts=None,
            crop_encoding=None,  # TODO remove if not needed in test stage
            checkpoint_epoch=None,  # TODO remove if not needed in test stage
            **kwargs,
    ):
        """
        Parameters:
        -----------
        linear_encoder: dict
            A dictionary mapping the true labels to the given labels.
            True labels = the labels in the mappings file.
            Given labels = labels ranging from 0 to len(true labels), which the
            true labels have been converted into.
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
        crop_encoding: dict, default None
            A dictionary mapping class ids to class names.
        checkpoint_epoch: int, default None
            The epoch loaded for testing.
        """
        super(BaseModelModule, self).__init__()
        self.save_hyperparameters(
            ignore=["class_counts", "linear_encoder", "crop_encoding", "bands", "num_time_steps"])

        self.linear_encoder = linear_encoder
        self.learning_rate = learning_rate
        self.parcel_loss = parcel_loss
        self.crop_encoding = crop_encoding
        self.checkpoint_epoch = checkpoint_epoch

        num_classes = len(set(linear_encoder.values()))

        # Loss function
        class_weights = self._calculate_class_weights(class_counts, parcel_loss).cuda() if weighted_loss else None

        self.loss_nll = nn.NLLLoss(weight=class_weights)
        self.loss_nll_parcel = nn.NLLLoss(weight=class_weights, ignore_index=0)
        self.metric_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.metric_acc_parcel = MulticlassAccuracy(num_classes=num_classes, average="macro", ignore_index=0)
        self.metric_f1w = MulticlassF1Score(num_classes=num_classes, average="weighted")
        self.metric_f1w_parcel = MulticlassF1Score(num_classes=num_classes, average="weighted", ignore_index=0)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize="none",
                                                          ignore_index=0 if parcel_loss else None)

        self.run_dir = Path(wandb.run.dir)

        self.model = get_model_class(model)(num_classes=num_classes, num_bands=len(bands), **kwargs)

        self.validation_examples = None

    def _calculate_class_weights(self, class_counts: dict[any, int], parcel_loss):
        """
        Calculate class weights for the loss function.
        """
        # Compute weights for each class
        filtered_counts = {
            self.linear_encoder[int(k)]:  # map the original label to the training label (for ordering)
                (v if (int(k) != 0 or not parcel_loss) else 0)  # if parcel_loss, ignore the zero class
            for k, v in class_counts.items()
            if int(k) in self.linear_encoder  # only include classes that are in the linear encoder
        }

        all_counts = sum(filtered_counts.values())
        n_classes = len(filtered_counts)
        class_weights = [
            (all_counts / (n_classes * v)) if v != 0 else 0  # if v == 0, then the class is ignored
            for k, v in sorted(filtered_counts.items())
        ]

        # Log class weights as a table to Wandb
        class_names = ["0"] + [name for _, name in SELECTED_CLASSES]
        class_weights_table = wandb.Table(
            columns=["ID", "Class Name", "Pixel count", "Weight"],
            data=[
                [id_, class_names[k], filtered_counts[k], class_weights[k]]
                for id_, k in sorted(self.linear_encoder.items())
                if (k != 0 or not parcel_loss)  # if parcel_loss, ignore the zero class
            ],
        )
        wandb.log({"class_weights": class_weights_table})

        return torch.tensor(class_weights)

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage == "validate":
            wandb.define_metric("val/acc", summary="max")
            wandb.define_metric("val/acc_parcel", summary="max")
            wandb.define_metric("val/f1w", summary="max")
            wandb.define_metric("val/f1w_parcel", summary="max")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        lr_scheduler: LRScheduler
        if self.hparams.get("model") == "unet":
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
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
            'monitor': self.hparams.get("monitor_metric"),
        }]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch['medians'], batch['labels']  # (B, T, C, H, W), (B, H, W)
        output = self.model(inputs)
        loss_nll = self.loss_nll(output, labels)
        loss_nll_parcel = self.loss_nll_parcel(output, labels)
        self.log('train/loss_nll', loss_nll,
                 on_step=True, on_epoch=True, logger=True, prog_bar=not self.parcel_loss)
        self.log('train/loss_nll_parcel', loss_nll_parcel,
                 on_step=True, on_epoch=True, logger=True, prog_bar=self.parcel_loss)

        loss = loss_nll_parcel if self.parcel_loss else loss_nll
        if torch.isnan(loss):
            return None
        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_examples = {
            "inputs": [],
            "labels": [],
            "outputs": [],
        }

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['medians'], batch['labels']  # (B, T, C, H, W), (B, H, W)
        output = self.model(inputs)
        has_examples, want_examples = len(self.validation_examples["inputs"]), 4
        if has_examples < want_examples:
            self.validation_examples["inputs"].extend(inputs[:(want_examples - has_examples)].cpu().detach().numpy())
            self.validation_examples["labels"].extend(labels[:(want_examples - has_examples)].cpu().detach().numpy())
            self.validation_examples["outputs"].extend(output[:(want_examples - has_examples)].cpu().detach().numpy())

        loss_nll = self.loss_nll(output, labels)
        loss_nll_parcel = self.loss_nll_parcel(output, labels)
        if not torch.isnan(loss_nll):
            self.log('val/loss_nll', loss_nll,
                     on_step=False, on_epoch=True, logger=True, prog_bar=not self.parcel_loss)
            self.log('val/loss_nll_parcel', loss_nll_parcel,
                     on_step=False, on_epoch=True, logger=True, prog_bar=self.parcel_loss)

        acc = self.metric_acc(output, labels)
        f1w = self.metric_f1w(output, labels)
        acc_parcel = self.metric_acc_parcel(output, labels)
        f1w_parcel = self.metric_f1w_parcel(output, labels)
        self.confusion_matrix.update(output, labels)
        self.log_dict({
            'val/acc': acc,
            'val/f1w': f1w,
        }, on_step=False, on_epoch=True, logger=True, prog_bar=not self.parcel_loss)
        self.log_dict({
            'val/acc_parcel': acc_parcel,
            'val/f1w_parcel': f1w_parcel
        }, on_step=False, on_epoch=True, logger=True, prog_bar=self.parcel_loss)

    def on_validation_epoch_end(self) -> None:
        confusion_matrix = self.confusion_matrix.compute()
        wandb_cm = self._get_wandb_confusion_matrix(confusion_matrix)

        examples_table = wandb.Table(columns=["id", "inputs", "ground truth", "predictions"])

        # TODO fix
        rgb_bands = ["B04", "B03", "B02"]
        images = []
        class_labels = {0: "0"} | {self.linear_encoder[k]: v for k, v in SELECTED_CLASSES}
        class_set = [
            {"id": id_, "name": name}
            for id_, name in class_labels.items()
        ]
        for i, (
                inputs,  # (T, C, H, W)
                labels,  # (H, W)
                outputs,  # (P, H, W)
        ) in enumerate(zip(
            self.validation_examples["inputs"],
            self.validation_examples["labels"],
            self.validation_examples["outputs"],
        )):
            pixels = np.flip(inputs[3, 0:3, :, :], axis=1).transpose((1, 2, 0))
            pixels_scaled = np.floor(pixels * 256).clip(0, 255).astype(np.uint8)

            images.append(wandb.Image(
                pixels_scaled,
                caption=f"Example {i + 1}",
                masks={
                    "ground_truth": {
                        "mask_data": labels,
                        "class_labels": class_labels,
                    },
                    "predictions": {
                        "mask_data": outputs.argmax(axis=0),
                        "class_labels": class_labels,
                    },
                },
                classes=class_set,
            ))

            examples_table.add_data(
                i,
                wandb.Image(pixels_scaled, caption=f"Example {i + 1}"),
                wandb.Image(
                    pixels_scaled,
                    masks={
                        "ground_truth": {
                            "mask_data": labels,
                            "class_labels": class_labels,
                        },
                    },
                ),
                wandb.Image(
                    pixels_scaled,
                    masks={
                        "predictions": {
                            "mask_data": outputs.argmax(axis=0),
                            "class_labels": class_labels,
                        },
                    },
                ),
            )

        if self.trainer.state.stage != "sanity_check":
            wandb.log({
                "epoch": self.current_epoch,
                "confusion_matrix": wandb_cm,
                "examples": images,
                "examples_table": examples_table,
            })

    def _get_wandb_confusion_matrix(self, confusion_matrix):
        class_names = ["0"] + [name for _, name in SELECTED_CLASSES]

        data = []
        for i in range(self.confusion_matrix.num_classes):
            if self.parcel_loss and i == 0:
                continue
            for j in range(self.confusion_matrix.num_classes):
                if self.parcel_loss and j == 0:
                    continue
                data.append([class_names[i], class_names[j], confusion_matrix[i, j]])

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

    def on_test_epoch_end(self):
        self.confusion_matrix = self.confusion_matrix.cpu().detach().numpy()

        self.confusion_matrix = self.confusion_matrix[1:, 1:]  # Drop zero label

        # Calculate metrics and confusion matrix
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tp = np.diag(self.confusion_matrix)
        tn = self.confusion_matrix.sum() - (fp + fn + tp)

        # Sensitivity, hit rate, recall, or true positive rate
        tpr = tp / (tp + fn)
        # Specificity or true negative rate
        tnr = tn / (tn + fp)
        # Precision or positive predictive value
        ppv = tp / (tp + fp)
        # Negative predictive value
        npv = tn / (tn + fn)
        # Fall out or false positive rate
        fpr = fp / (fp + tn)
        # False negative rate
        fnr = fn / (tp + fn)
        # False discovery rate
        fdr = fp / (tp + fp)
        # F1-score
        f1 = (2 * ppv * tpr) / (ppv + tpr)

        # Overall accuracy
        accuracy = (tp + tn) / (tp + fp + fn + tn)

        # Export metrics in text file
        metrics_file = self.run_dir / f"evaluation_metrics_epoch{self.checkpoint_epoch}.csv"

        # Delete file if present
        metrics_file.unlink(missing_ok=True)

        with open(metrics_file, "a") as f:
            row = 'Class'
            for k in sorted(self.linear_encoder.keys()):
                if k == 0: continue
                row += f',{k} ({self.crop_encoding[k]})'
            f.write(row + '\n')

            row = 'tn'
            for i in tn:
                row += f',{i}'
            f.write(row + '\n')

            row = 'tp'
            for i in tp:
                row += f',{i}'
            f.write(row + '\n')

            row = 'fn'
            for i in fn:
                row += f',{i}'
            f.write(row + '\n')

            row = 'fp'
            for i in fp:
                row += f',{i}'
            f.write(row + '\n')

            row = "specificity"
            for i in tnr:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = "precision"
            for i in ppv:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = "recall"
            for i in tpr:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = "accuracy"
            for i in accuracy:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = "f1"
            for i in f1:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = 'weighted macro-f1'
            class_samples = self.confusion_matrix.sum(axis=1)
            weighted_f1 = ((f1 * class_samples) / class_samples.sum()).sum()
            f.write(row + f',{weighted_f1:.4f}\n')

        # Normalize each row of the confusion matrix because class imbalance is
        # high and visualization is difficult
        row_mins = self.confusion_matrix.min(axis=1)
        row_maxs = self.confusion_matrix.max(axis=1)
        cm_norm = (self.confusion_matrix - row_mins[:, None]) / (row_maxs[:, None] - row_mins[:, None])

        # Export Confusion Matrix

        # Replace invalid values with 0
        self.confusion_matrix = np.nan_to_num(self.confusion_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        sns.heatmap(self.confusion_matrix, annot=False, ax=ax, cmap="Blues", fmt="g")

        # Labels, title and ticks
        label_font = {'size': '18'}
        ax.set_xlabel('Predicted labels', fontdict=label_font, labelpad=10)
        ax.set_ylabel('Observed labels', fontdict=label_font, labelpad=10)

        ax.set_xticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))
        ax.set_yticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.set_xticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0],
                           fontsize=8, rotation='vertical')
        ax.set_yticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0],
                           fontsize=8, rotation='horizontal')

        ax.tick_params(axis='both', which='major')

        title_font = {'size': '21'}
        ax.set_title('Confusion Matrix', fontdict=title_font)

        for i in range(len(self.linear_encoder.keys()) - 1):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))

        plt.savefig(self.run_dir / f'confusion_matrix_epoch{self.checkpoint_epoch}.png', dpi=fig.dpi,
                    bbox_inches='tight', pad_inches=0.5)

        np.save(self.run_dir / f'cm_epoch{self.checkpoint_epoch}.npy', self.confusion_matrix)

        # Export normalized Confusion Matrix

        # Replace invalid values with 0
        cm_norm = np.nan_to_num(cm_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        sns.heatmap(cm_norm, annot=False, ax=ax, cmap="Blues", fmt="g")

        # Labels, title and ticks
        label_font = {'size': '18'}
        ax.set_xlabel('Predicted labels', fontdict=label_font, labelpad=10)
        ax.set_ylabel('Observed labels', fontdict=label_font, labelpad=10)

        ax.set_xticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))
        ax.set_yticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.set_xticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0],
                           fontsize=8, rotation='vertical')
        ax.set_yticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0],
                           fontsize=8, rotation='horizontal')

        ax.tick_params(axis='both', which='major')

        title_font = {'size': '21'}
        ax.set_title('Confusion Matrix', fontdict=title_font)

        for i in range(len(self.linear_encoder.keys()) - 1):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))

        plt.savefig(self.run_dir / f'confusion_matrix_norm_epoch{self.checkpoint_epoch}.png', dpi=fig.dpi,
                    bbox_inches='tight', pad_inches=0.5)

        np.save(self.run_dir / f'cm_norm_epoch{self.checkpoint_epoch}.npy', self.confusion_matrix)
        pickle.dump(self.linear_encoder, open(self.run_dir / f'linear_encoder_epoch{self.checkpoint_epoch}.pkl', 'wb'))
