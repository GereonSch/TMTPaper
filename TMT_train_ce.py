import torch
import torch.nn as nn
# import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.classification import Precision, Recall, F1Score, BinarySpecificity


class CrossEntropyModule(pl.LightningModule):

    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-4, label_smoothing=0.0):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.save_hyperparameters(ignore=["model"])

        # Metrics (binary classification)
        self.train_precision = Precision('binary', num_classes=2, average='macro')
        self.train_recall = Recall('binary', num_classes=2, average='macro')
        self.train_f1 = F1Score('binary', num_classes=2, average='macro')

        self.val_precision = Precision('binary', num_classes=2, average='macro')
        self.val_recall = Recall('binary', num_classes=2, average='macro')
        self.val_f1 = F1Score('binary', num_classes=2, average='macro')
        self.val_specificity = BinarySpecificity()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, subj_labels, class_labels = batch  # subject_labels are not used
        labels = class_labels

        logits = self.model(x)
        loss = self.loss_fn(logits, labels)

        # preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)  # only for metrics / inference
        preds = torch.argmax(probs, dim=1)

        # Compute metrics
        acc = (preds == labels).float().mean()
        precision = self.train_precision(preds, labels)
        recall = self.train_recall(preds, labels)
        f1 = self.train_f1(preds, labels)

        # logging metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_precision", precision, prog_bar=False)
        self.log("train_recall", recall, prog_bar=False)
        self.log("train_f1", f1, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, subj_labels, class_labels = batch  # subject_labels are not used
        labels = class_labels
        logits = self.model(x)
        loss = self.loss_fn(logits, labels)

        # preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)  # only for metrics / inference
        preds = torch.argmax(probs, dim=1)

        # Compute metrics
        acc = (preds == labels).float().mean()
        precision = self.val_precision(preds, labels)
        recall = self.val_recall(preds, labels)
        f1 = self.val_f1(preds, labels)
        specificity = self.val_specificity(preds, labels)

        # Logging metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_precision", precision, prog_bar=False)
        self.log("val_recall", recall, prog_bar=False)
        self.log("val_specificity", specificity, prog_bar=False)
        self.log("val_f1", f1, prog_bar=False)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Add this to Reduce LR when val_loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            # factor=0.5,      # reduce LR by half
            # patience=5,      # wait 5 epochs with no improvement
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            #verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # must match the metric logged during validation
                "interval": "epoch",
                "frequency": 1
            },
        }
