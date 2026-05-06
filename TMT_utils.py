import pandas as pd
import numpy as np
import os, yaml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import torch
import pytorch_lightning as pl


# ===============================================
# Load tabulated data and optionally
# apply nomrlization, PCA or ICA
# ===============================================
def load_config(fname_cfg):
    # load config file
    with open(fname_cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    # dataset
    if isinstance(cfg['data']['filenames'], list):
        dataset = 'TMT-A/B'
    else:
        dataset = "TMT-A" if "TMTA" in cfg['data']['filenames'] else "TMT-B"
    cfg['data'].update(dataset=dataset)

    # Check seed and set generator
    SEED = cfg.get('seed', None)
    generator = torch.Generator()
    if (SEED):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        pl.seed_everything(SEED, workers=True)
        generator.manual_seed(SEED)
        is_deterministic = True
    else:
        is_deterministic = False
    cfg.update(is_deterministic=is_deterministic)
    cfg.update(generator=generator)

    # device cuda or cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.update(device=device)

    # logging
    cfg['use_wandb'] = cfg.get('use_wandb', False)

    return cfg


# ===============================================
# Load and combine tabulated TMT datasets
# optionally apply normalization, PCA, or ICA
# Note: returns only unique subjects
# ===============================================
def load_TMT(config):
    # ------------------------------
    # handle INPUT
    # ------------------------------
    filenames = [os.path.join(config['data']['path'], f) for f in config['data']['filenames']]
    if isinstance(filenames, str):
        # Convert single filename to a list for unified processing
        fnames = [filenames]
    else:
        fnames = filenames
    all_dfs = []
    drop_col = config['data']['drop_columns']
    do_norm = config['data']['do_norm']
    norm_axis = config['data']['norm_axis']
    n_comp_pca = config['data']['n_comp_pca']
    n_comp_ica = config['data']['n_comp_ica']
    base_columns = None  # Define a base set of columns based on the first file for standardization

    # ------------------------------
    # Loop over files
    # ------------------------------
    for i, file in enumerate(fnames):
        if not os.path.exists(file):  # check if file exist
            print(f"Warning: File not found: {file}. Skipping.")
            continue

        # check format: cvs or xlsx (default)
        name, ext = file.split('.')[-2:]
        if ext == 'csv':
            df_orig = pd.read_csv(file, header=0, sep=';')  # csv format: has NO subject IDs
        else:
            header = 1 if name[-4:] == 'TMTA' else 0
            df_orig = pd.read_excel(file, header=header)  # excel format: has subject IDs
        df_cleaned = df_orig.dropna(how='any')  # Drop rows with any NaNs

        # drop columns if they exist
        df_processed = df_cleaned.drop(
            columns=[col for col in drop_col if col in df_cleaned.columns],
            errors='ignore'  # Safely ignore columns that don't exist
        )

        # Standardize column names (we assume all files have the same columns)
        if i == 0:
            # Set the column names of the first dataset as the target
            base_columns = df_processed.columns.tolist()
        else:
            # For subsequent datasets, check if the columns match the base.
            if len(df_processed.columns) == len(base_columns):
                # This assumes columns are in the correct order but named differently.
                # If column order is NOT guaranteed, you need a more complex mapping (not shown).
                df_processed.columns = base_columns
            else:
                print(
                    f"Warning: Columns in file {file} do not match the expected number. Skipping standardization for this file.")
        all_dfs.append(df_processed)


    # ------------------------------
    # concatenate all data frames
    # ------------------------------
    if not all_dfs:
        return np.array([]), np.array([])
    df_combined = pd.concat(all_dfs, ignore_index=True)

    # ------------------------------
    # extract X and y and subjects
    # ------------------------------
    try:
        # create a mask to keep subjects that appear only one time
        subject_series = df_combined.loc[:, df_combined.columns.str.startswith('Psych_Code')].iloc[:, 0]
        subject_counts = subject_series.value_counts()
        subjects_to_keep = subject_counts[subject_counts > 1].index.tolist()
        df_filtered = df_combined[subject_series.isin(subjects_to_keep)]

        # Group labels (y)
        mask_label = df_filtered.columns.str.startswith('Gruppe')
        y = df_filtered.loc[:, mask_label].to_numpy().squeeze()

        # Subjects (subjects) - now only containing those with duplicates
        mask_subject = df_filtered.columns.str.startswith('Psych_Code')
        subjects = df_filtered.loc[:, mask_subject].to_numpy().squeeze()

        # Features (X)
        mask_to_exclude = df_filtered.columns.str.startswith('Psych_Code') | \
                          df_filtered.columns.str.startswith('Gruppe')
        mask_features = ~mask_to_exclude
        X = df_filtered.loc[:, mask_features].to_numpy()

        # Maskiere Features (Spalten, die nicht Gruppe oder Subject sind)
        mask_to_exclude = df_filtered.columns.str.startswith('Psych_Code') | df_filtered.columns.str.startswith(
            'Gruppe')
        feature_names = df_filtered.loc[:, ~mask_to_exclude].columns.tolist()

        # Update config
        config['data']['feature_names'] = feature_names

    except KeyError:
        raise ValueError("The final combined DataFrame must contain a 'Gruppe' column for labels.")

    # ------------------------------
    # normalize data
    # ------------------------------
    if do_norm:
        scaler = StandardScaler()
        if norm_axis == 0:  # normalisation across subjects
            scaler.fit(X)
            X = scaler.transform(X)
        else:  # normalisation across parameters
            scaler.fit(X.T)
            X = scaler.transform(X.T).T

    # ------------------------------
    # apply PCA or ICA
    # ------------------------------
    if n_comp_ica > 0:
        ica = FastICA(n_components=n_comp_ica)
        X = ica.fit_transform(X)
    elif n_comp_pca > 0:
        pca = PCA(n_components=n_comp_pca)
        X = pca.fit_transform(X)

    # ------------------------------
    # update config
    # ------------------------------
    # shapes
    n_row, n_col = X.shape
    if (config['augmentation']['crop'] > 0) and (config['augmentation']['crop'] < n_col):
        n_col_train = n_col - config['augmentation']['crop']
    else:
        config['augmentation'].update(crop=0)
        n_col_train = n_col
    # batch size
    if config['train']['batch_size'] < 4:
        batch_size = n_row
    else:
        batch_size = config['train']['batch_size']

    # update config
    config['data'].update(
        n_row=n_row,
        n_col=n_col,
        n_col_train=n_col_train,
        crop=config['augmentation']['crop'],
        n_subjects=len(np.unique(subjects)),
    )

    # update batch size
    config['train'].update(batch_size=batch_size)

    return X, y, subjects


# ========================================================
# LSO mask
# ========================================================
def get_lso_mask_paired(array_of_subjects, ratio_test=0.2, n_splits=10, seed=None):
    """
    Create Leave-Subject-Out (LSO) cross-validation masks.
    We assume that for each subject only exactly two different recordings exist.

    For each random split, a subset of subjects is held out (used for validation),
    and the remaining are used for training. The function returns a list of test_masks

    Parameters:
    -----------
    array_of_subjects : array-like
        1D array where each element corresponds to the subject ID of a data sample.
        Multiple entries may have the same subject ID.

    num_splits : int
        Number of folds to divide the unique subjects into (default is 10).

    seed : int or None
        Optional random seed for reproducible shuffling of subjects.

    Returns:
    --------
    test_masks : list of np.ndarray (bool)
        List of boolean masks, each indicating which samples belong to the validation
        set for that split. Complement is implicitly the training set.
    """

    # Identify all unique subject IDs from the dataset
    unique_subjects = np.unique(array_of_subjects)
    n_subjects = len(unique_subjects)

    # For each split randomly shuffle the subject IDs to ensure randomness in fold assignment
    test_masks = []
    for isplit in range(n_splits):
        if seed:
            seed += isplit

        # Initialize a local RNG (reproducible and isolated)
        rng = np.random.default_rng(seed)
        idx_perm = rng.permutation(n_subjects)

        # shuffle unique subjects
        subjects = unique_subjects[idx_perm]

        # get subset for testing
        test_subj = subjects[:int(ratio_test * n_subjects)]  # select the first n% of subjects for testing
        mask = np.isin(array_of_subjects, test_subj)
        test_masks.append(mask)  # append mask

    if n_splits == 1:
        test_masks = test_masks[0]

    return test_masks


# ============================================
# compute metrics
# ============================================
def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # disp = ConfusionMatrixDisplay(confusion_matrix=cmn, display_labels=["Above_Median", "Below_Median"])
    # disp.plot(cmap=plt.cm.Oranges)

    TP = cmn[1][1]
    TN = cmn[0][0]
    FP = cmn[0][1]
    FN = cmn[1][0]
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    # classification report
    report = classification_report(y_true, y_pred)

    metrics = dict(
        accuracy=acc,
        sensitivity=sensitivity,
        specificity=specificity,
        TP=TP,
        TN=TN,
        FP=FP,
        FN=FN,
        report=report,
    )

    return metrics


# ============================================
# Data augmentation:
# - add channels by permuting values in columns
# ============================================
def augmentation_add_chan(X_orig, add_chan=100, seed=None):
    """
    Data augmentation by permuting column values to create additional channels.

    Parameters
    ----------
    X_orig : np.ndarray
        Input array of shape (n_subj, n_col)
    add_chan : int, optional
        Number of channels to generate, by default 100
    seed : int, optional
        Seed for reproducible random permutations
    """
    n_subj, n_col = X_orig.shape  # must be two dimensional

    # Initialize a local RNG (reproducible and isolated)
    rng = np.random.default_rng(seed)

    if add_chan > 1:
        X = np.empty((n_subj, add_chan, n_col))
        X[:, 0, :] = X_orig  # original data as first channel

        # Each new channel gets a distinct permutation of columns
        for i in range(1, add_chan):
            permutation_indices = rng.permutation(n_col)
            # Apply the same permutation across all subjects
            X[:, i, :] = X_orig[:, permutation_indices]
    else:
        return X_orig

    return X


