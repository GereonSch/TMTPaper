#################################################
#
# DL-based TMT Classification using combined TMT A/B
#
#################################################
import numpy as np
import time
import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os

os.makedirs("logging", exist_ok=True)

from TMT_utils import load_TMT, augmentation_add_chan, load_config
from TMT_utils_MPL import load_model, get_dataloaders



#################################################
# Load SETTINGS via config
#################################################
# Load configuration
fname_cfg = 'config_LRO-CE.yaml'  # LRO with ContrastiveLossg
#fname_cfg = 'config_LSO-TPL.yaml'      # LSO with TripletLoss
cfg = load_config(fname_cfg)

# ====================================tupel (ce und tpl). was
# Load data and combine datasets
# ====================================
X_orig, y_np, subjects = load_TMT(cfg)
#feature_names = X_orig.columns.tolist()

# ==============================
# Loop over repititions
# ====================================
start_time = time.time()
results = []
acc_all = []
acc_list = []
checkpoint_paths = [] #neu
test_sets = [] #neu
device = "cuda" if torch.cuda.is_available() else "cpu" #neu
torch.cuda.empty_cache()
for irepeat in range(cfg['train']['n_repeat']):

    cfg.update(irep=irepeat)

    if cfg['is_deterministic']:
        seed = cfg['seed'] + 10
        cfg.update(seed=seed)  # get a fixed seed for each repetition

    # =========================
    # Data Augmentation
    # add additional channels
    # for each subject additional permuted column values are added
    # =========================
    X_np = augmentation_add_chan(X_orig, add_chan=cfg['augmentation']['add_chan'], seed=cfg['seed'])

    # ====================================
    # Get DataLoaders
    # ====================================
    train_loader, test_loader = get_dataloaders(X_np, y_np, subjects, cfg)

    # ====================================
    # load and init model
    # ====================================
    # TODO: add additional head for triplet loss config to allow classification on both, subject and class_labels
    model = load_model(cfg)

    # ====================================
    # Set up Wandb logger
    # ====================================
    if cfg['use_wandb']:
        wandb_logger = WandbLogger(
            project=cfg['wandb']['project'],
            entity=cfg['wandb']['entity'],
            tags=cfg['wandb']['notes'],
            mode='online',
            dir='logging',
            config=cfg,
        )
    else:
        wandb_logger = None

    from pytorch_lightning.loggers import CSVLogger

    csv_logger = CSVLogger(
        save_dir="lightning_logs",
        name="tmt_cv"
    )

    # ====================================
    # Early stopping
    # ====================================
    early_stop = EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        min_delta=0.0,  # Minimum change to qualify as an improvement
        patience=25,  # Number of epochs with no improvement after which training will be stopped
        mode='min',  # 'min' for loss, 'max' for accuracy/F1, etc.
        verbose=True,
    )

    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, verbose=True, filename=f"rep_{irepeat}")   # Änderung hier

    # ====================================
    # trainer
    # ====================================
    trainer = pl.Trainer(
        max_epochs=cfg['train']['max_epochs'],
        #logger=wandb_logger,
        logger=[wandb_logger, csv_logger] if cfg['use_wandb'] else csv_logger,
        accelerator=cfg['trainer']['accelerator'],
        devices=cfg['trainer']['device'],
        log_every_n_steps=1,
        gradient_clip_val=cfg['trainer']['gradient_clip_val'],  # clip gradients to stabilize updates
        # callbacks=[early_stop, checkpoint],                         # uncomment to use early stopping
        callbacks=[checkpoint],  #geändert
        deterministic=cfg['is_deterministic'],
    )

    for i, x in enumerate(test_sets):
        print(i, np.mean(x))

    # ====================================
    # fit model
    # ====================================
    trainer.fit(model, train_loader, test_loader)

    # finish logging
    if cfg['use_wandb']:
        wandb.finish()  # finish logging for this split and repitition

    # ====================================
    # collect ACC
    # ====================================
    acc = trainer.logged_metrics['val_acc']
    acc_list.append(acc)

    # store one test batch per repetition for later SHAP
    test_batch = next(iter(test_loader))
    X_test_np = test_batch[0].cpu().numpy()
    test_sets.append(X_test_np)  # neu
    np.save(f"logging/test_batch_rep={irepeat}.npy", X_test_np) #notwendig?
    checkpoint_paths.append(checkpoint.best_model_path) #neu

# ====================================
# collect metrics
# TODO: make sure to collect the best and not the last metric value
# ====================================
acc = np.array(acc_list)
print('--------------------------')
print()
print('Dataset: %s' % cfg['data']['dataset'])
print('Seed:    %s' % 'random' if not cfg['seed'] else 'fixed')
print('Number of repetitions: %d' % cfg['train']['n_repeat'])
print('   Accs:  ' + ', '.join(['%.3f' % val for val in acc]))
print('   min:   ', acc.min())
print('   max:   ', acc.max())
print('   mean:  ', acc.mean())
print('   std:   ', acc.std())
print('   median:', np.median(acc))
# TODO: add the following metrics also for triplet loss
if (cfg['model']['loss_type'] == 'CrossEntropy'):
    print('Precision:            % .3f' % trainer.logged_metrics['val_precision'])
    print('Specificity:          % .3f' % trainer.logged_metrics['val_specificity'])
    print('Sensitivity (recall): % .3f' % trainer.logged_metrics['val_recall'])
    print('F1:                   % .3f' % trainer.logged_metrics['val_f1'])
print('--------------------------')
print()
print('>>> DONE !')

#Shap-Analyse
import shap
import torch
import matplotlib.pyplot as plt

background_idx = np.random.choice(
    X_orig.shape[0],
    size=min(100, X_orig.shape[0]),
    replace=False
)
background = X_orig[background_idx, :]

all_shap_values = []
all_x_eval = []

class_index = 1  # positive Klasse

for ckpt_path, X_eval in zip(checkpoint_paths, test_sets):

    model = load_model(cfg)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(
        state["state_dict"] if "state_dict" in state else state
    )

    model.to(device)
    model.eval()

    def model_predict(x_np):
        x_torch = torch.from_numpy(x_np).float().to(device)

        with torch.no_grad():
            logits = model(x_torch)
            probs = torch.nn.functional.softmax(
                logits,
                dim=1
            ).cpu().numpy()

        return probs

    explainer = shap.KernelExplainer(
        model_predict,
        background
    )

    shap_vals = explainer.shap_values(X_eval)

    shap_vals = np.array(shap_vals)

    # erwartet: (n_samples, n_features, n_classes)
    shap_vals_class = shap_vals[:, :, class_index]

    all_shap_values.append(shap_vals_class)
    all_x_eval.append(X_eval)

# Alle Holdouts zusammenführen
shap_vals_global = np.vstack(all_shap_values)
X_eval_global = np.vstack(all_x_eval)

print("Global X shape:", X_eval_global.shape)
print("Global SHAP shape:", shap_vals_global.shape)

assert shap_vals_global.shape == X_eval_global.shape

shap.summary_plot(
    shap_vals_global,
    X_eval_global,
    feature_names=cfg['data']['feature_names'],
    show=True
)

print("Shap plots done!")

# ====================================
# WandB Plots: val_loss und val_acc über Epochen
# ====================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

log_root = "lightning_logs/tmt_cv"
versions = sorted(
    d for d in os.listdir(log_root)
    if d.startswith("version_")
)

all_val_losses = []
all_val_accs = []
all_epochs_list = []

for v in versions:
    csv_path = os.path.join(log_root, v, "metrics.csv")
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)

    # Je nach PL-Version heißen die Spalten z. B. "epoch", "val_loss", "val_acc"
    if "val_loss" not in df.columns or "val_acc" not in df.columns:
        continue

    # nur Zeilen mit val_loss/val_acc (PL schreibt manchmal NaNs auf anderen Steps)
    df_val = df.dropna(subset=["val_loss", "val_acc"])
    if "epoch" in df_val.columns:
        epochs = df_val["epoch"].values
    else:
        epochs = np.arange(len(df_val))

    val_loss = df_val["val_loss"].values
    val_acc = df_val["val_acc"].values

    all_val_losses.append(val_loss)
    all_val_accs.append(val_acc)
    all_epochs_list.append(epochs)

if len(all_val_losses) == 0:
    raise RuntimeError("Keine val_loss/val_acc Daten in CSV-Logs gefunden.")

max_epochs = min(len(l) for l in all_val_losses)

val_losses_arr = np.array([l[:max_epochs] for l in all_val_losses])
val_accs_arr  = np.array([a[:max_epochs] for a in all_val_accs])
epochs_arr    = np.arange(max_epochs)

mean_loss = val_losses_arr.mean(axis=0)
std_loss  = val_losses_arr.std(axis=0)
mean_acc  = val_accs_arr.mean(axis=0)
std_acc   = val_accs_arr.std(axis=0)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(epochs_arr, mean_loss, label="val_loss (mean)", color="C0")
ax[0].fill_between(epochs_arr, mean_loss - std_loss, mean_loss + std_loss, color="C0", alpha=0.3)
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_title("Validation Loss (mean ± std)")
ax[0].legend()

ax[1].plot(epochs_arr, mean_acc, label="val_acc (mean)", color="C1")
ax[1].fill_between(epochs_arr, mean_acc - std_acc, mean_acc + std_acc, color="C1", alpha=0.3)
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].set_title("Validation Accuracy (mean ± std)")
ax[1].legend()

plt.tight_layout()


plt.show()
print("Done with Validation Curves")