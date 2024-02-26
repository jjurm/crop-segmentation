import pickle
from abc import ABC, abstractmethod
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

from models.convstar import ConvSTAR
from models.unet import UNet


def get_model_class(self, model_name):
    if model_name == "unet":
        model_class = UNet
    elif model_name == "convstar":
        model_class = ConvSTAR
    else:
        raise ValueError(f"model = {model_name}, expected: 'unet' or 'convstar'")
    return model_class


class BaseModelModule(pl.LightningModule, ABC):
    def __init__(
            self,
            model_name: str,
            linear_encoder,
            learning_rate,
            parcel_loss=False,
            class_weights=None,
            crop_encoding=None,
            checkpoint_epoch=None,  # TODO remove if not needed in test stage
            **kwargs
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
        parcel_loss: boolean, default False
            If True, then a custom loss function is used which takes into account
            only the pixels of the parcels. If False, then all image pixels are
            used in the loss function.
        class_weights: dict, default None
            Weights per class to use in the loss function.
        crop_encoding: dict, default None
            A dictionary mapping class ids to class names.
        checkpoint_epoch: int, default None
            The epoch loaded for testing.
        num_layers: int, default 3
            The number of layers to use in each path.
        """
        super(BaseModelModule, self).__init__()
        self.save_hyperparameters()

        self.linear_encoder = linear_encoder
        self.learning_rate = learning_rate
        self.parcel_loss = parcel_loss
        self.crop_encoding = crop_encoding
        self.checkpoint_epoch = checkpoint_epoch

        num_discrete_labels = len(set(linear_encoder.values()))
        self.confusion_matrix = torch.zeros([num_discrete_labels, num_discrete_labels])

        # Loss function
        class_weights = torch.tensor(
            [class_weights[k] for k in sorted(class_weights.keys())]).cuda() if class_weights is not None else None
        self.lossfunction = nn.NLLLoss(
            ignore_index=0,
            reduction='sum' if self.parcel_loss else 'mean',
            weight=class_weights,
        )

        self.run_dir = Path(wandb.run.dir)

        model_class = get_model_class(model_name)
        self.model = model_class(num_discrete_labels=num_discrete_labels, **kwargs)

        self.epoch_train_losses: list = None
        self.epoch_valid_losses: list = None

    @abstractmethod
    def get_lr_scheduler(self, optimizer):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        lr_scheduler: LRScheduler = None
        if self.hparams.get("model_name") == "unet":
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                factor=0.5,
                patience=4,
                verbose=True
            )
        elif self.hparams.get("model_name") == "convstar":
            lr_scheduler = StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )

        return [optimizer], [{
            'scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }]

    def loss(self, batch):
        inputs = batch['medians']
        label = batch['labels'].to(torch.long)  # (B, H, W)

        pred = self.model(inputs)  # (B, K, H, W)

        if self.parcel_loss:
            parcels = batch['parcels']  # (B, H, W)
            parcels_K = parcels[:, None, :, :].repeat(1, pred.size(1), 1, 1)  # (B, K, H, W)

            # Note: a new masked array must be created in order to avoid inplace
            # operations on the label/pred variables. Otherwise the optimizer
            # will throw an error because it requires the variables to be unchanged
            # for gradient computation

            mask = (parcels) & (label != 0)
            mask_K = (parcels_K) & (label[:, None, :, :].repeat(1, pred.size(1), 1, 1) != 0)

            label_masked = label.clone()
            label_masked[~mask] = 0

            pred_masked = pred.clone()
            pred_masked[~mask_K] = 0

            label = label_masked.clone()
            pred = pred_masked.clone()

            loss = self.lossfunction(pred, label)

            loss = loss / parcels.sum()
        else:
            loss = self.lossfunction(pred, label)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.epoch_train_losses.append(loss.item() * batch['medians'].shape[0])
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.epoch_valid_losses.append(loss.item() * batch['medians'].shape[0])
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        inputs = batch['medians']
        label = batch['labels'].to(torch.long)  # (B, H, W)

        pred = self.model(inputs).to(torch.long)  # (B, K, H, W)

        # Reverse the logarithm of the LogSoftmax activation
        pred = torch.exp(pred)

        # Clip predictions larger than the maximum possible label
        pred = torch.clamp(pred, 0, max(self.linear_encoder.values()))

        if self.parcel_loss:
            parcels = batch['parcels']  # (B, H, W)
            parcels_K = parcels[:, None, :, :].repeat(1, pred.size(1), 1, 1)  # (B, K, H, W)

            mask = (parcels) & (label != 0)
            mask_K = (parcels_K) & (label[:, None, :, :].repeat(1, pred.size(1), 1, 1) != 0)
            label[~mask] = 0
            pred[~mask_K] = 0

            pred_sparse = pred.argmax(axis=1)

            label = label.flatten()
            pred = pred_sparse.flatten()

            # Discretize predictions
            # bins = np.arange(-0.5, sorted(list(self.linear_encoder.values()))[-1] + 0.5, 1)
            # bins_idx = torch.bucketize(pred, torch.tensor(bins).cuda())
            # pred_disc = bins_idx - 1

        for i in range(label.shape[0]):
            self.confusion_matrix[label[i], pred[i]] += 1

        return

    def on_train_epoch_start(self) -> None:
        self.epoch_train_losses = []

    def on_train_epoch_end(self) -> None:
        # Calculate average loss over an epoch
        train_loss = np.nanmean(self.epoch_train_losses)

        with open(self.run_dir / "avg_train_losses.txt", 'a') as f:
            f.write(f'{self.current_epoch}: {train_loss}\n')

        with open(self.run_dir / 'lrs.txt', 'a') as f:
            f.write(f'{self.current_epoch}: {self.learning_rate}\n')

        self.log('train_loss', train_loss, prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        self.epoch_valid_losses = []

    def on_validation_epoch_end(self) -> None:
        # Calculate average loss over an epoch
        valid_loss = np.nanmean(self.epoch_valid_losses)

        with open(self.run_dir / "avg_val_losses.txt", 'a') as f:
            f.write(f'{self.current_epoch}: {valid_loss}\n')

        self.log('val_loss', valid_loss, prog_bar=True)

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
