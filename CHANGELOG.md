# Changelog

All notable changes to the TMT classification pipeline will be documented in this file.

## TMT_main_v3.py

### Added
- **True Determinism:** Added `pytorch_lightning.seed_everything(seed, workers=True)` inside the training loop to ensure PyTorch, NumPy, and Python random number generators are fully and correctly seeded for every repetition.
- **Configurable SHAP Explainer:** Updated the SHAP analysis to respect a new `shap:` configuration block in `config_LRO-CE.yaml`. The explainer (`KernelExplainer`, `PermutationExplainer`, `DeepExplainer`), `max_evals`, `nsamples`, and `batch_size` can now be set dynamically.

### Changed
- **Data Leakage Fix (Checkpoints):** Disabled test set optimization by changing `ModelCheckpoint` to `save_last=True`, `save_top_k=0`, and `monitor=None`. The model no longer monitors the test data (`val_loss`) to pick the "best" epoch, preventing data leakage during hyper-parameter selection.
- **Unbiased Metric Collection:** Metric collection (`val_acc`) now correctly extracts the accuracy from the final epoch's logs, rather than cherry-picking the "best" checkpoint performance.
- **Robust SHAP Batch Collection:** Test and train batches for SHAP analysis are now properly aggregated across all batches (`torch.cat([...])`) instead of just parsing the first batch (`next(iter(loader))`). This makes SHAP robust against `batch_size` changes.

### Fixed
- **PermutationExplainer IndexError:** Fixed a bug in `shap`'s `PermutationExplainer` that threw an `IndexError` when the evaluation batch size was smaller than the `batch_size` parameter. Moved the `batch_size` argument out of the `__init__` constructor and passed it directly during the evaluation call `explainer(..., batch_size=...)`.

## TMT_main_v2.py

### Fixed
- **DeepExplainer Crash:** Fixed `shap.DeepExplainer` failing due to incompatible argument types. It requires a PyTorch graph, so the generic Python wrapper `model_predict` was replaced with the actual PyTorch `model` (`nn.Module`), and input data was converted to PyTorch `tensors` instead of NumPy arrays.
- **Missing Plot Directories:** Corrected the `plt.savefig` paths for the SHAP summary and bar plots. The previous hardcoded paths (`shapts/summary_plot.png` and `shap_plots/bar_plot.png`) were causing runtime errors. They now correctly point to `logging/shap_plots/`, which is created dynamically during runtime.