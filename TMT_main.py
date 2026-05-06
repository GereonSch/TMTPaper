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

    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, verbose=True)

    # ====================================
    # trainer
    # ====================================
    trainer = pl.Trainer(
        max_epochs=cfg['train']['max_epochs'],
        logger=wandb_logger,
        accelerator=cfg['trainer']['accelerator'],
        devices=cfg['trainer']['device'],
        log_every_n_steps=1,
        gradient_clip_val=cfg['trainer']['gradient_clip_val'],  # clip gradients to stabilize updates
        # callbacks=[early_stop, checkpoint],                         # uncomment to use early stopping
        # callbacks=[checkpoint],
        deterministic=cfg['is_deterministic'],
    )

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

# Nach allen 10 Läufen
all_test_sets = []  # Sammle Testdaten
for irepeat in range(cfg['train']['n_repeat']):
    ...
    _, test_loader = get_dataloaders(X_np, y_np, subjects, cfg)
    test_batch = next(iter(test_loader))
    all_test_sets.append(test_batch[0].cpu().numpy())

# Mittelwert über alle Wiederholungen
test_samples = np.mean(np.stack(all_test_sets, axis=0), axis=0)



model.eval()

# ===============================
# 1️⃣ Vorbereitungen: Hilfsfunktion für Vorhersagen
# ===============================
def model_predict(x_np):
    x_torch = torch.from_numpy(x_np).float().to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(x_torch)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
    return probs

# ===============================
# 2️⃣ Hintergrund-Daten für KernelExplainer
# Hier am besten aus Trainingsdaten, z.B. X_np
background = X_np[np.random.choice(X_np.shape[0], 100, replace=False)]




# ===============================
# 4️⃣ Explainer erstellen
explainer = shap.KernelExplainer(model_predict, background)

# ===============================
# 5️⃣ SHAP Werte berechnen für echten Test-Batch
shap_values = explainer.shap_values(test_samples)

print(f"Feature count laut Config: {len(cfg['data']['feature_names'])}")
print(f"Test sample shape: {test_samples[0].shape}")
print(f"SHAP-Wert shape: {np.array(shap_values).shape}")

# ===============================
# 6️⃣ SHAP Werte für Klasse 1 extrahieren (bei binärer Klassifikation)
class_index = 1
shap_vals_class1 = shap_values[:, :, class_index]

# ===============================
# 7️⃣ Visualisierung

# Globaler Überblick: summary plot
shap.summary_plot(shap_vals_class1, test_samples, feature_names=cfg['data']['feature_names'])

# Force plot für das erste Beispiel
shap.force_plot(
    explainer.expected_value[class_index],
    shap_vals_class1[0],
    test_samples[0],
    feature_names=cfg['data']['feature_names']
)



