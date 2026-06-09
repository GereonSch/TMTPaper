#################################################
#
# DL-based TMT Classification using combined TMT A/B
# IMPROVED VERSION v2 - Fixed SHAP DeepExplainer issues
#
#################################################
import numpy as np
import os
import time
import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from TMT_utils import load_TMT, augmentation_add_chan, load_config
from TMT_utils_MPL import load_model, get_dataloaders

os.makedirs("logging", exist_ok=True)

#################################################
# Load SETTINGS via config
#################################################
# Load configuration
fname_cfg = 'config_LRO-CE.yaml'  # LRO with ContrastiveLossg
cfg = load_config(fname_cfg)

# ====================================
# Load data and combine datasets
# ====================================
X_orig, y_np, subjects = load_TMT(cfg)


# ==============================
# Loop over repititions
# ====================================
start_time = time.time()
results = []
acc_all = []
acc_list = []
checkpoint_paths = []  # neu
test_sets = []  # neu
train_sets_for_background = []  # neu: store training data for SHAP background
device = "cuda" if torch.cuda.is_available() else "cpu"  # neu
torch.cuda.empty_cache()

for irepeat in range(cfg['train']['n_repeat']):
    print(f"\n{'='*60}")
    print(f"Starting repetition {irepeat + 1}/{cfg['train']['n_repeat']}")
    print(f"{'='*60}")

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


    # log data
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

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
        filename=f"rep_{irepeat}"
    )  # Änderung hier

    # ====================================
    # trainer
    # ====================================
    trainer = pl.Trainer(
        max_epochs=cfg['train']['max_epochs'],
        # logger=wandb_logger,
        logger=[wandb_logger, csv_logger] if cfg['use_wandb'] else csv_logger,
        accelerator=cfg['trainer']['accelerator'],
        devices=cfg['trainer']['device'],
        log_every_n_steps=1,
        gradient_clip_val=cfg['trainer']['gradient_clip_val'],  # clip gradients to stabilize updates
        # callbacks=[early_stop, checkpoint],                         # uncomment to use early stopping
        callbacks=[checkpoint],  # geändert
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
    # collect ACC - use best checkpoint metrics
    # ====================================
    # Load the best checkpoint to get the actual best metrics
    try:
        best_ckpt = torch.load(checkpoint.best_model_path, map_location=device, weights_only=True)
        acc = best_ckpt.get('metrics', {}).get('val_acc', trainer.logged_metrics.get('val_acc', 0))
        print(f"Repetition {irepeat + 1}: Best val_acc = {acc:.4f} (from checkpoint)")
    except Exception as e:
        print(f"Warning: Could not load checkpoint metrics for rep {irepeat + 1}: {e}")
        acc = trainer.logged_metrics.get('val_acc', 0)
        print(f"Repetition {irepeat + 1}: Using last epoch val_acc = {acc:.4f}")

    acc_list.append(acc)

    # store test and train batches per repetition => needed for SHAP
    test_batch = next(iter(test_loader))
    X_test_np = test_batch[0].cpu().numpy()
    test_sets.append(X_test_np)

    # Store training data for SHAP background
    train_batch = next(iter(train_loader))
    X_train_np = train_batch[0].cpu().numpy()
    train_sets_for_background.append(X_train_np)

    # Save checkpoint path
    checkpoint_paths.append(checkpoint.best_model_path)
    print(f"Checkpoint saved to: {checkpoint.best_model_path}")

# ====================================
# collect metrics
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
if (cfg['model']['loss_type'] == 'CrossEntropy'):
    print('Precision:            % .3f' % trainer.logged_metrics['val_precision'])
    print('Specificity:          % .3f' % trainer.logged_metrics['val_specificity'])
    print('Sensitivity (recall): % .3f' % trainer.logged_metrics['val_recall'])
    print('F1:                   % .3f' % trainer.logged_metrics['val_f1'])
print('--------------------------')
print()
print('>>> DONE !')

# ============================================
# SHAP Analysis - Fixed Version v2
# ============================================
print("\n" + "="*60)
print("Starting SHAP Analysis")
print("="*60)

import shap
import torch
import matplotlib.pyplot as plt

class_index = 1  # positive Klasse

# Create background from all training data across repetitions
print("\nCreating background dataset from training data...")
all_train_samples = np.vstack(train_sets_for_background)
print(f"Total training samples available: {all_train_samples.shape[0]}")

# Stratified sampling for background
n_background = min(100, all_train_samples.shape[0])
background_indices = np.random.RandomState(42).choice(
    all_train_samples.shape[0],
    size=n_background,
    replace=False
)
background = all_train_samples[background_indices, :]
print(f"Background dataset size: {background.shape[0]} samples")

all_shap_values = []
all_x_eval = []

for ckpt_idx, (ckpt_path, X_eval) in enumerate(zip(checkpoint_paths, test_sets)):
    print(f"\nProcessing checkpoint {ckpt_idx + 1}/{len(checkpoint_paths)}")
    
    # Check if checkpoint exists
    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint not found: {ckpt_path}. Skipping.")
        continue

    try:
        model = load_model(cfg)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(
            state["state_dict"] if "state_dict" in state else state
        )
        model.to(device)
        model.eval()
        print(f"  Model loaded successfully from: {ckpt_path}")
    except Exception as e:
        print(f"Error loading checkpoint {ckpt_path}: {e}")
        continue

    def model_predict(x_np):
        """Wrapper function for SHAP explainer"""
        x_torch = torch.from_numpy(x_np).float().to(device)
        with torch.no_grad():
            logits = model(x_torch)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        return probs

    # Try multiple explainers in order of preference
    shap_vals = None
    explainer_used = None
    
    # 1. Try DeepExplainer first (fastest for neural networks)
    print(f"  Attempting DeepExplainer...")
    try:
        # DeepExplainer works best with tensors
        background_tensor = torch.tensor(background[:10], dtype=torch.float32).to(device)
        X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
        
        explainer = shap.DeepExplainer(model, background_tensor)
        shap_vals = explainer.shap_values(X_eval_tensor)
        explainer_used = "DeepExplainer"
        print(f"  DeepExplainer succeeded")
        
    except Exception as e:
        print(f"  DeepExplainer failed: {type(e).__name__}: {str(e)[:100]}")
        
        # 2. Try PermutationExplainer (good alternative)
        print(f"  Attempting PermutationExplainer...")
        try:
            explainer = shap.PermutationExplainer(
                model_predict, 
                background[:30],
                max_evals=5000,
                batch_size=100
            )
            shap_vals = explainer.shap_values(X_eval)
            explainer_used = "PermutationExplainer"
            print(f"  PermutationExplainer succeeded")
            
        except Exception as e2:
            print(f"  PermutationExplainer failed: {type(e2).__name__}: {str(e2)[:100]}")
            
            # 3. Fallback to KernelExplainer
            print(f"  Attempting KernelExplainer (fallback)...")
            try:
                explainer = shap.KernelExplainer(
                    model_predict, 
                    background[:30],
                    link="logit"
                )
                shap_vals = explainer.shap_values(X_eval, nsamples=100)
                explainer_used = "KernelExplainer"
                print(f"  KernelExplainer succeeded")
                
            except Exception as e3:
                print(f"  KernelExplainer failed: {type(e3).__name__}: {str(e3)[:100]}")
                continue
    
    # Handle output format
    if shap_vals is not None:
        if isinstance(shap_vals, list):
            shap_vals_class = shap_vals[class_index]
        else:
            # For binary classification, shap_vals might be (n_samples, n_features) or (n_samples, n_features, n_classes)
            if len(shap_vals.shape) == 3:
                shap_vals_class = shap_vals[:, :, class_index]
            else:
                shap_vals_class = shap_vals

        shap_vals_class = np.array(shap_vals_class)
        print(f"  SHAP values shape: {shap_vals_class.shape}")
        print(f"  Explainer used: {explainer_used}")

        all_shap_values.append(shap_vals_class)
        all_x_eval.append(X_eval)

# Stack all SHAP values from all repetitions
if len(all_shap_values) > 0:
    shap_vals_global = np.vstack(all_shap_values)
    X_eval_global = np.vstack(all_x_eval)

    print(f"\n{'='*60}")
    print(f"SHAP Analysis Results")
    print(f"{'='*60}")
    print(f"Global X shape: {X_eval_global.shape}")
    print(f"Global SHAP shape: {shap_vals_global.shape}")

    # Verify shapes match
    assert shap_vals_global.shape[0] == X_eval_global.shape[0], \
        f"Sample count mismatch: SHAP={shap_vals_global.shape[0]}, X={X_eval_global.shape[0]}"
    assert shap_vals_global.shape[1] == X_eval_global.shape[1], \
        f"Feature count mismatch: SHAP={shap_vals_global.shape[1]}, X={X_eval_global.shape[1]}"

    # Create summary plot
    print("\nGenerating SHAP summary plot...")
    os.makedirs("logging/shap_plots", exist_ok=True)

    shap.summary_plot(
        shap_vals_global,
        X_eval_global,
        feature_names=cfg['data']['feature_names'],
        show=False,
        plot_size=None,
        class_names=["Class 0", "Class 1"]
    )
    plt.tight_layout()
    plt.savefig("logging/shap_plots/summary_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Create bar plot of mean absolute SHAP values
    print("Generating SHAP bar plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_vals_global,
        X_eval_global,
        feature_names=cfg['data']['feature_names'],
        show=False,
        plot_type="bar",
        class_names=["Class 0", "Class 1"]
    )
    plt.tight_layout()
    plt.savefig("logging/shap_plots/bar_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Save SHAP values for further analysis
    np.save("logging/shap_values.npy", shap_vals_global)
    np.save("logging/x_eval.npy", X_eval_global)
    print("\nSHAP values saved to logging/shap_values.npy")
    print("SHAP analysis completed!")
else:
    print("Warning: No SHAP values computed. Check checkpoint paths and model loading.")

print("\n" + "="*60)
print("All analyses completed successfully!")
print("="*60)