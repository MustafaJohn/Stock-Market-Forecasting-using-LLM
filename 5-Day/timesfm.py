# %%
import numpy as np
import pandas as pd
import torch
from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# %%
# Parameters and settings

# Parameters for data split
WINDOW = 5  # rolling window size to use as predictors (for TimesFM window size should be multiple of 32 - input patch length)
DATE_COL = 'DlyCalDt'
ID_COL = 'PERMNO'
TARGET_COL = 'ExcessReturn'

# Estimation (in sample) period dates
in_sample_start_date = pd.to_datetime("2000-01-01")
in_sample_end_date = pd.to_datetime("2015-12-31")

# Out-of-sample period dates
out_sample_start_date = pd.to_datetime("2016-01-01")
out_sample_end_date = pd.to_datetime("2024-12-31")

# Use GPU if available, else default to using CPU
device_map = "cpu" 
device = torch.device("cpu")

# %%
# Load the cleaned and filtered data files for in sample and out of sample periods into a pandas DataFrames
in_sample_df = pd.read_csv('/mnt/iusers01/msc-stu/hum-msc-data-sci-2024-2025/d14099as/scratch/thesis/in_sample_cleaned.csv')
out_sample_df = pd.read_csv('/mnt/iusers01/msc-stu/hum-msc-data-sci-2024-2025/d14099as/scratch/thesis/out_sample_cleaned.csv')


# Ensure the date columns are in datetime format
in_sample_df[DATE_COL] = pd.to_datetime(in_sample_df[DATE_COL])
out_sample_df[DATE_COL] = pd.to_datetime(out_sample_df[DATE_COL])

in_sample_df = in_sample_df[[ID_COL, DATE_COL, TARGET_COL]]
out_sample_df = out_sample_df[[ID_COL, DATE_COL, TARGET_COL]]

in_sample_df = in_sample_df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)
out_sample_df = out_sample_df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

# %%
in_sample_df.info()
out_sample_df.info()

# %%
stocks_permno = in_sample_df["PERMNO"].unique().tolist()
print(f"Number of unique stocks: {len(stocks_permno)}")

# %%
# Create rolling window for predictors for Bolt models

combined_df = pd.concat([in_sample_df, out_sample_df])
combined_df = combined_df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)
combined_df[DATE_COL] = pd.to_datetime(combined_df[DATE_COL])

contexts = []
targets = []
records = []

for id, grp in combined_df.groupby(ID_COL):
    values = grp[TARGET_COL].values
    dates = grp[DATE_COL].values

    for i in range(len(values) - WINDOW):
        pred_date = dates[i + WINDOW]
        if pred_date >= pd.to_datetime("2016-01-01"):
            contexts.append(torch.tensor(values[i:i+WINDOW], dtype=torch.float32, device=device))
            targets.append(values[i+WINDOW])
            records.append({
                ID_COL: id,
                TARGET_COL: values[i+WINDOW],
                DATE_COL: pred_date
            })

# %%
print(len(contexts))

# %%
y_test = pd.Series(targets)

results = pd.DataFrame(records)

# %%
# Creating a Function to Calculate Predictive-R2 Used in the Finance Literature
def r2(y_true, y_pred):
    return 1-(((y_true-y_pred)**2).sum()/(y_true**2).sum())

# Directional Accuracy Split
def directional_accuracy(y_true, y_pred):
    sign_match = np.sign(y_true) == np.sign(y_pred)
    up_da = sign_match[y_true > 0].mean() if np.any(y_true > 0) else np.nan
    down_da = sign_match[y_true < 0].mean() if np.any(y_true < 0) else np.nan
    return up_da, down_da

# %%
# Zero Shot TimesFM-1.0-200M
tfm1 = TimesFm(
    hparams = TimesFmHparams(
        context_len = 32,
        horizon_len = 1,
        input_patch_len = 32,
        output_patch_len = 128,
        num_layers = 20,
        model_dims = 1280,
        backend = device_map
        ),
    checkpoint = TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch")
    )
freqs = [0] * len(contexts)
preds, _ = tfm1.forecast(contexts, freq=freqs)

y_tfm1 = pd.Series(preds.reshape([-1,]))

results['y_tfm1'] = y_tfm1
r2_tfm1  = r2(y_test, y_tfm1)
mse_tfm1 = mean_squared_error(y_test, y_tfm1)
mae_tfm1 = mean_absolute_error(y_test, y_tfm1)
da_tfm1 = (np.sign(y_test) == np.sign(y_tfm1)).mean()
up_da_tfm1, down_da_tfm1 = directional_accuracy(y_test, y_tfm1)

# %%
# Zero Shot TimesFM-2.0-500M
tfm2 = TimesFm(
    hparams = TimesFmHparams(
        context_len = 32,
        horizon_len = 1,
        input_patch_len = 32,
        output_patch_len = 128,
        num_layers = 50,
        model_dims = 1280,
        backend = device_map
        ),
    checkpoint = TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch")
    )

freqs = [0] * len(contexts)
preds, _ = tfm2.forecast(contexts, freq=freqs)

y_tfm2 = pd.Series(preds.reshape([-1,]))

results['y_tfm2'] = y_tfm2
r2_tfm2  = r2(y_test, y_tfm2)
mse_tfm2 = mean_squared_error(y_test, y_tfm2)
mae_tfm2 = mean_absolute_error(y_test, y_tfm2)
da_tfm2 = (np.sign(y_test) == np.sign(y_tfm2)).mean()
up_da_tfm2, down_da_tfm2 = directional_accuracy(y_test, y_tfm2)

# %%
# Collating Results

results_matrix = [{
        "Model": "TimesFM-1.0-200M",
        "R-squared": r2_tfm1,
        "MSE": mse_tfm1,
        "MAE": mae_tfm1,
        "Direction Accuracy": da_tfm1,
        "Up Directional Accuracy": up_da_tfm1,
        "Down Directional Accuracy": down_da_tfm1
    },
    {
        "Model": "TimesFM-2.0-500M",
        "R-squared": r2_tfm2,
        "MSE": mse_tfm2,
        "MAE": mae_tfm2,
        "Direction Accuracy": da_tfm2,
        "Up Directional Accuracy": up_da_tfm2,
        "Down Directional Accuracy": down_da_tfm2
    }]

results_matrix_df = pd.DataFrame(results_matrix)
results_matrix_df.to_csv("timesfm(5-day)_results.csv", index=False)

# %%
# Save Prediction Results
results.to_csv("timesfm(5-day)_predictions.csv", index=False)

