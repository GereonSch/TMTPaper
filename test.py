import pandas as pd

df_a = pd.read_excel("./data/ExecPapTMTA.xlsx")
df_b = pd.read_excel("./data/ExecPapTMTB.xlsx")



print(f"Anzahl feature_names: {len(cfg['data']['feature_names'])}")
print(f"Form test_samples[0]: {test_samples[0].shape}")  # Sollte (num_features,) sein
print(f"Form shap_values[1][0]: {shap_values[1][0].shape}")  # Sollte (num_features,) sein
