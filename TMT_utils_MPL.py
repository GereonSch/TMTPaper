import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from TMT_train_ce import CrossEntropyModule
from TMT_train_tpl import MetricLearningModule
from TMT_utils import get_lso_mask_paired



# ========================================================
# Dataset class for TMT data
# ========================================================
class Dataset_TMT(Dataset):
    """
    Dataset that returns (x, subj_label, class_label).

    Args:
        X: numpy array or tensor shape (N, D)
        subj_labels: array-like of integer subject IDs (dtype: int)
        class_labels: array-like of integer class labels (dtype: int) - e.g. 0/1 or 1/2
        device: optional device to put tensors on (e.g., "cuda")
        transform: optional callable applied to x
    """

    def __init__(self, X, subj_labels, class_labels, device=None, transform=None):
        self.X = X
        # assume subj_labels and class_labels are already integer arrays
        self.subj = torch.tensor(subj_labels, dtype=torch.long)
        self.clazz = torch.tensor(class_labels, dtype=torch.long)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        subj = self.subj[idx]
        clazz = self.clazz[idx]

        # apply transform (if any). The transforms should accept numpy arrays or tensors.
        if self.transform:
            x = self.transform(x)

        # ensure x is float32 tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.to(dtype=torch.float32)

        # move to device if provided
        if self.device:
            x = x.to(self.device)
            subj = subj.to(self.device)
            clazz = clazz.to(self.device)

        return x, subj, clazz


# ========================================================
# Transform Pipeline to apply cropping
# ========================================================
# def transform_permutation(seed: int):
def transform_permutation(seed=None):
    """
    Returns a callable that applies permuting the data

    Args:
        n_samples: number of n_samples used (or -1 for using all n_samples).
        random_crop: True for random subwindow, False for deterministic full/leading window.
    """
    return TransformPermute(seed=seed)


# ========================================================
# Class to apply permutation
# ========================================================
class TransformPermute:
    """
    shuffle the data across samples

    Args:
        seed

    """

    def __init__(self, seed=None):
        self.seed = seed

    def process_permute(self, x, idx_permute):
        # Ensure x has at least 2 dimensions
        x_processed = np.atleast_2d(x)

        # Use the existing slice. '...' selects all preceding dimensions.
        #    If x was (10, 305), x_processed is (10, 305) -> Slice works normally.
        #    If x was (305), x_processed is (1, 305) -> Slice on (1, 305) works.
        subset = x_processed[..., idx_permute]

        # to return the original shape for 1D input, we have to np.squeeze
        if x.ndim == 1:
            return np.squeeze(subset)

        return subset

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) > 1:  # [n_chan, n_samples]
            n_chan, n_samples = x.shape
        else:
            n_chan = 1
            n_samples = x.shape[-1]
        self.n_chan = n_chan
        self.n_samples = n_samples

        # Initialize a local RNG (reproducible and isolated)
        rng = np.random.default_rng(self.seed)
        idx_perm = rng.permutation(n_samples)

        # Apply the same permutation across all channels
        x = self.process_permute(x, idx_perm)

        return x


# ========================================================
# Transform Pipeline to apply cropping
# ========================================================
def transform_cropping(n_samples: int, crop: int, random_crop: bool):
    """
    Returns a callable that applies cropping n_samples

    Args:
        n_samples: number of n_samples used (or -1 for using all n_samples).
        random_crop: True for random subwindow, False for deterministic full/leading window.
    """
    return TransformCrop(n_samples=n_samples, crop=crop, random_crop=random_crop)


# ========================================================
# Class to apply cropping
# ========================================================
class TransformCrop:
    """
    return a subwindow of data (cropped by n_samples)

    Args:
        n_samples: number of n_samples used (or -1 for using all n_samples).
        random_crop: if True, pick a random sub-window; else use start at 0 (full window).
    """

    def __init__(self, n_samples: int, crop: int, random_crop: bool = True):
        self.n_samples = n_samples
        self.crop = crop
        self.min_samples = 5  # we set the minimum number of samples to 5
        self.random_crop = random_crop
        if (crop > n_samples - self.min_samples) or (crop < 1):
            self.crop = 0  # use all samples
            self.win_samples = n_samples
        else:
            self.win_samples = n_samples - crop

    def process_crop(self, x, start, win_samples):
        # Ensure x has at least 2 dimensions
        x_processed = np.atleast_2d(x)

        # Use the existing slice. '...' selects all preceding dimensions.
        #    If x was (10, 305), x_processed is (10, 305) -> Slice works normally.
        #    If x was (305), x_processed is (1, 305) -> Slice on (1, 305) works.
        subset = x_processed[..., start: start + win_samples]

        # to return the original shape for 1D input, we have to np.squeeze
        if x.ndim == 1:
            return np.squeeze(subset)

        return subset

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x shape [n_roi, n_times]

        if (self.win_samples >= self.min_samples) and (self.win_samples < self.n_samples):
            max_start = self.n_samples - self.win_samples
            start = random.randint(0, max_start) if self.random_crop else 0
            x = self.process_crop(x, start, self.win_samples)

        return x


# ========================================================
# Wrapper to get the transformer
# ========================================================
def get_transformer(config):
    """Selects the augmentation pipeline based on config."""
    if config['augmentation']['permute']:
        transformer = transform_permutation(seed=config['seed'])
    elif config['augmentation']['crop'] > 0:
        transformer = transform_cropping(
            config['data']['n_col'], config['augmentation']['crop'], random_crop=True
        )
    else:
        transformer = None
    return transformer


# ========================================================
# MLP Base Model
# ========================================================
class BaseEmbeddingNet(nn.Module):
    def __init__(self, input_dim, dim_out=32, bn_last_layer=True, *,
                 loss_type='CrossEntropy', n_classes=2, dropout_rate=0.0):
        super().__init__()
        self.bn_last_layer = bn_last_layer
        self.loss_type = loss_type
        self.dropout_rate = dropout_rate

        self.flatten = nn.Flatten()

        # Layer 1
        self.fc1 = nn.Linear(input_dim, 2 * dim_out)
        # self._init_weights(self.fc1)
        self.bn1 = nn.BatchNorm1d(2 * dim_out, momentum=0.01, eps=0.001)
        self.dropout1 = nn.Dropout(dropout_rate)  # <== set to 0.0 for no dropout

        # Layer 2
        self.fc2 = nn.Linear(2 * dim_out, dim_out)
        # self._init_weights(self.fc2)
        if bn_last_layer:
            self.bn2 = nn.BatchNorm1d(dim_out, momentum=0.01, eps=0.001)
        self.dropout2 = nn.Dropout(dropout_rate / 2)  # # <== set to 0.0 for no dropout

        if loss_type == 'CrossEntropy':
            self.classifier = nn.Linear(dim_out, n_classes)

    def forward(self, x):
        x = self.flatten(x)

        # Layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        if self.bn_last_layer:
            x = self.bn2(x)
        x = self.dropout2(x)

        if self.loss_type == 'CrossEntropy':
            x = self.classifier(x)
            # NOTE: PyTorch’s nn.CrossEntropyLoss expects raw logits as input — not probabilities,
            # because it internally applies a log_softmax before computing the loss.
            # x = F.softmax(x, dim=1)   # do NOT apply softmax here
        else:  # Triplet/Circle loss: L2-normalize embeddings
            x = F.normalize(x, p=2, dim=1)

        return x


# ========================================================
# Load Model wrapper
# ========================================================
def load_model(cfg):
    #print(">>> LOSS TYPE:", cfg['model'].get('loss_type'))
    Xshape = cfg['data']['shape_Xtrain']
    n_samples = cfg['data']['n_col_train']
    output_dim = cfg['model']['dim_out']
    learning_rate = cfg['model'].get('lr', 1e-4)
    dropout = cfg['model'].get('dropout', 0.0)
    weight_decay = cfg['model'].get('weight_decay', 0.0)
    label_smoothing = cfg['model'].get('label_smoothing', 0.0)
    loss_type = cfg['model'].get('loss_type', 'triplet')
    n_classes = cfg['model'].get('n_classes', 2)


    input_dim = Xshape[1] * n_samples if len(Xshape) == 3 else n_samples

    if loss_type == 'CrossEntropy':
        # CrossEntropy model
        ce_net = BaseEmbeddingNet(
            input_dim=input_dim,
            dim_out=output_dim,
            n_classes=n_classes,
            dropout_rate=dropout,
            loss_type=loss_type
        )
        model = CrossEntropyModule(
            model=ce_net,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
        )

    else:
        # Other: triplet, contrastive or circle
        embedding_model = BaseEmbeddingNet(
            input_dim=input_dim,
            dim_out=output_dim,
            n_classes=n_classes,
            dropout_rate=dropout,
            loss_type=loss_type
        )
        cfg['model'].update(n_classes=cfg['data']['n_subjects'])
        model = MetricLearningModule(embedding_model, cfg)

    return model


# ========================================================
# Helper to build train/test DataLoaders for metric learning
# ========================================================
def get_dataloaders(X_np, y_np, subjects, cfg):
    """
    Prepares train/test DataLoaders for metric learning (triplet, contrastive, circle).
    Encodes subject strings -> integers, and class labels -> 0/1.
    """

    # Encode subject IDs
    le_subj = LabelEncoder().fit(subjects)
    subj_ids_all = le_subj.transform(subjects)

    # Encode class labels (convert 1/2 to 0/1 if needed)
    unique_classes = np.unique(y_np)
    class_map = {c: i for i, c in enumerate(sorted(unique_classes))}
    class_ids_all = np.array([class_map[v] for v in y_np], dtype=np.int64)

    # Create train/test masks
    if cfg['train']['mode'] == "LSO":
        # mask_test = get_lso_mask_paired(subjects, n_splits=cfg['train']['n_repeat'], seed=cfg['seed'])
        mask_test = get_lso_mask_paired(subjects, n_splits=1, seed=cfg['seed'])
        mask_train = ~mask_test
    else:  # LRO
        sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg['data']['ratio_val'], random_state=cfg['seed'])
        train_idx, test_idx = next(sss.split(X_np, class_ids_all))
        mask_train = np.zeros(len(X_np), dtype=bool)
        mask_train[train_idx] = True
        mask_test = ~mask_train

    # Split arrays
    Xtrain = X_np[mask_train]
    Xtest = X_np[mask_test]
    subj_train = subj_ids_all[mask_train]
    subj_test = subj_ids_all[mask_test]
    class_train = class_ids_all[mask_train]
    class_test = class_ids_all[mask_test]
    cfg['data'].update(shape_Xtrain=Xtrain.shape, shape_Xtest=Xtest.shape)

    # Transformer
    transformer = get_transformer(cfg)

    # Datasets
    train_ds = Dataset_TMT(Xtrain, subj_train, class_train, device=cfg['device'], transform=transformer)
    test_ds = Dataset_TMT(Xtest, subj_test, class_test, device=cfg['device'], transform=transformer)

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['trainer']['num_workers'],
        generator=cfg['generator'],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['trainer']['num_workers'],
    )

    return train_loader, test_loader