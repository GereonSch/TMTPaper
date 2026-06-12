import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import wandb
import pandas as pd

from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import CustomKNN


# ==============================
# Metric Learning LightningModule
# ==============================
class MetricLearningModule(pl.LightningModule):
    """
    PyTorch Lightning module for metric learning (Triplet, NTXent, Circle).

    Training uses subject IDs for loss computation.
    Validation uses class labels (e.g., healthy vs non-healthy) for accuracy evaluation.
    """

    def __init__(self, model, cfg):
        super().__init__()

        self.model = model
        self.cfg = cfg

        # -------------------------
        # CONFIG
        # -------------------------
        self.loss_type = cfg["model"].get("loss_type", "triplet")
        self.lr = cfg["model"].get("lr", 1e-4)
        self.gamma = cfg["model"].get("gamma", 80)
        self.margin = cfg["model"].get("margin", 0.2)
        self.triplet_type = cfg["model"].get("triplet_type", "semihard")

        distance_type = cfg["model"].get("distance_metric", "CosineSimilarity")
        self.distance_metric = (
            CosineSimilarity()
            if distance_type == "CosineSimilarity"
            else LpDistance(normalize_embeddings=True, p=2)
        )

        # -------------------------
        # LOSS & MINER
        # -------------------------
        if self.loss_type == "triplet":
            self.loss_fn = losses.NTXentLoss(temperature=0.1)
            self.miner = miners.BatchEasyHardMiner(
                pos_strategy='hard',
                neg_strategy='semihard',
                # neg_strategy='all',        # XXX: is to hard, needs more data
                allowed_pos_range=None,
                allowed_neg_range=None,
            )

        elif self.loss_type == "NTXent":
            self.loss_fn = losses.NTXentLoss(temperature=0.1)
            self.miner = None

        elif self.loss_type == "circle":
            self.loss_fn = losses.CircleLoss(
                m=0.25,  # Not the same as margin parameter
                gamma=self.gamma,
                distance=self.distance_metric
            )
            self.miner = None  # CircleLoss handles pairs internally

        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        # -------------------------
        # ACCURACY CALCULATORS
        # -------------------------
        self.knn_func = CustomKNN(distance=self.distance_metric, batch_size=None)

        self.accuracy_calculator = AccuracyCalculator(
            include=["precision_at_1"],
            k="max_bin_count",
            device=self.device,
            return_per_class=False,
            knn_func=self.knn_func,
        )

        self.per_class_calculator = AccuracyCalculator(
            include=["precision_at_1"],
            k="max_bin_count",
            device=self.device,
            return_per_class=True,
            knn_func=self.knn_func,
        )

    # -------------------------
    # FORWARD
    # -------------------------
    def forward(self, x):
        return self.model(x)

    # -------------------------
    # TRAINING STEP
    # -------------------------
    def training_step(self, batch, batch_idx):
        x, subj_labels, class_labels = batch  # subject_labels are not used
        labels = class_labels
        emb = self.model(x)

        if self.loss_type == "circle":
            loss = self.loss_fn(emb, labels)
        else:
            triplets = self.miner(emb, labels)
            loss = self.loss_fn(emb, labels, triplets)

        with torch.no_grad():
            tm = self.accuracy_calculator.get_accuracy(
                query=emb,
                query_labels=labels,
                reference=emb,
                reference_labels=labels,
                ref_includes_query=True
            )
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", tm["precision_at_1"], on_epoch=True, prog_bar=True)

        return loss

    # -------------------------
    # VALIDATION
    # -------------------------
    def on_validation_epoch_start(self):
        self._val_embeddings = []
        self._val_labels = []
        self._val_losses = []

    def validation_step(self, batch, batch_idx):
        x, subj_labels, class_labels = batch  # subject_labels are not used
        labels = class_labels
        emb = self.model(x)

        if self.loss_type == "circle":
            val_loss = self.loss_fn(emb, labels)
        else:
            triplets = self.miner(emb, labels)
            val_loss = self.loss_fn(emb, labels, triplets)

        self._val_losses.append(val_loss)
        self._val_embeddings.append(emb)
        self._val_labels.append(labels)

    def on_validation_epoch_end(self):
        # Skip if no data (sanity check)
        if not self._val_embeddings:
            self._val_losses.clear()
            return

        # Aggregate
        avg_val_loss = torch.stack(self._val_losses).mean()
        all_emb = torch.cat(self._val_embeddings, dim=0)
        all_lbl = torch.cat(self._val_labels, dim=0)

        # Overall accuracy
        overall = self.accuracy_calculator.get_accuracy(
            query=all_emb,
            query_labels=all_lbl,
            reference=all_emb,
            reference_labels=all_lbl,
            ref_includes_query=True
        )
        self.log("val_loss", avg_val_loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", overall["precision_at_1"], on_epoch=True, prog_bar=True)

        # Only log at the final epoch
        if self.current_epoch == self.trainer.max_epochs - 1:
            per_class = self.per_class_calculator.get_accuracy(
                query=all_emb,
                query_labels=all_lbl,
                reference=all_emb,
                reference_labels=all_lbl,
                ref_includes_query=True
            )
            per_list = per_class.get("precision_at_1", [])
            if per_list:
                labels = torch.unique(all_lbl).cpu().tolist()
                # Build and sort DataFrame
                df = pd.DataFrame({
                    "class": labels,
                    "accuracy": [float(a) for a in per_list]
                }).sort_values("accuracy", ascending=True).reset_index(drop=True)
                # Log sorted table
                table = wandb.Table(dataframe=df)
                wandb.log({"val_class_performance": table})

        # Clear buffers
        self._val_embeddings.clear()
        self._val_labels.clear()
        self._val_losses.clear()


# -------------------------
    # TRAINING EPOCH END
    # -------------------------
    def on_train_epoch_end(self):
        if not hasattr(self, "_train_embeddings") or not self._train_embeddings:
            return

        all_emb = torch.cat(self._train_embeddings)
        all_lbl = torch.cat(self._train_labels)
        avg_loss = torch.stack(self._train_losses).mean()

        # Compute epoch-level training accuracy across all samples
        tm = self.accuracy_calculator.get_accuracy(
            query=all_emb,
            query_labels=all_lbl,
            reference=all_emb,
            reference_labels=all_lbl,
            ref_includes_query=False  # exclude self-matches for true retrieval accuracy
        )

        self.log("train_acc", tm["precision_at_1"], prog_bar=True)
        self.log("train_loss_epoch", avg_loss, prog_bar=True)

        # Clear accumulated tensors for next epoch
        self._train_embeddings.clear()
        self._train_labels.clear()
        self._train_losses.clear()

    # -------------------------
    # OPTIMIZER
    # -------------------------
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, eps=1e-8)
