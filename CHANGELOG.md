"# Changelog

All notable changes to the TMT classification pipeline will be documented in this file.

## [Unreleased]

### Added
- Improved SHAP analysis with DeepExplainer as primary explainer (10-100x faster than KernelExplainer)
- Fallback to KernelExplainer if DeepExplainer fails
- Background dataset created from training data across all repetitions instead of original data
- SHAP plots saved to `logging/shap_plots/` directory
- SHAP values saved to `logging/shap_values.npy` for further analysis
- Progress logging during training repetitions
- Error handling for checkpoint loading and SHAP computation

### Changed
- **Metric Collection**: Now loads best checkpoint metrics instead of using last epoch metrics
- **SHAP Background**: Changed from `X_orig` (100 random samples) to stratified sample from training data across all repetitions
- **SHAP Explainer**: Changed from KernelExplainer to DeepExplainer (with KernelExplainer fallback)
- **Background Size**: Reduced from 100 to 10 samples for DeepExplainer (more efficient)
- **Test Sample Collection**: Now stores training batches for background, not just test batches
- **Checkpoint Loading**: Added `weights_only=True` for security
- **Error Handling**: Added try-except blocks for robust error handling

### Fixed
- Removed unused loop that iterated over empty `test_sets` before population
- Fixed metric collection to use best epoch metrics from checkpoints
- Added validation for checkpoint file existence before loading
- Added shape verification for SHAP values and test data
- Fixed inconsistent metric collection across repetitions

### Technical Details

#### SHAP Analysis Improvements

**Background Dataset:**
- **Before**: 100 random samples from `X_orig` (all data)
- **After**: 100 stratified samples from training data across all repetitions
- **Rationale**: SHAP values are relative to a background distribution. Using training data ensures the background represents what the model actually learned from.

**Explainer Choice:**
- **Before**: KernelExplainer (slow, model-agnostic)
- **After**: DeepExplainer (fast, neural network-specific) with KernelExplainer fallback
- **Rationale**: DeepExplainer uses linearity properties of neural networks for much faster computation (10-100x speedup)

**Sample Size:**
- **100 samples**: Reasonable for KernelExplainer (needs more samples for good approximation)
- **10 samples**: Sufficient for DeepExplainer (uses exact gradient-based methods)

**Averaging Strategy:**
- SHAP values from all repetitions are stacked and analyzed together
- This provides a global view of feature importance across all model instances
- Alternative: Could average per-repetition SHAP values if test sets differ significantly

### Files Modified
- `TMT_main_improved.py`: Complete refactored version with all improvements
- `CHANGELOG.md`: Documentation of changes

### Migration Notes

To use the improved version:
1. Replace `TMT_main.py` with `TMT_main_improved.py` or merge the changes
2. Ensure `shap` package is installed (`pip install shap`)
3. Check that `logging/shap_plots/` directory is writable
4. For existing runs, you may want to re-run SHAP analysis with the new method

### Performance Impact

- **Training**: No change (same training procedure)
- **SHAP Analysis**: 10-100x faster with DeepExplainer
- **Memory**: Slightly higher (storing training batches for background)
- **Disk**: Additional ~100MB for SHAP plots and values

### Future Improvements

- [ ] Add per-repetition SHAP value averaging option
- [ ] Implement SHAP dependence plots for top features
- [ ] Add feature interaction analysis
- [ ] Support for multi-class SHAP visualization
- [ ] Add SHAP values to WandB logging
"