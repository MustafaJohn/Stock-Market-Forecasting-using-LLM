# %%
import pandas as pd
import numpy as np
import torch
from chronos import BaseChronosPipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tqdm as notebook_tqdm

# %%
# Using GPU if available
device_map = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Load Chronos model
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small",  # or "chronos-t5-small"
    device_map=device,
    torch_dtype=torch.float32
)

# %%
# Constants
ID_COL = "PERMNO"
TARGET_COL = "ExcessReturn"
DATE_COL = "DlyCalDt"
WINDOW = 21  # days of context

# Load data
in_sample_df = pd.read_csv('/mnt/iusers01/msc-stu/hum-msc-data-sci-2024-2025/d14099as/scratch/thesis/in_sample_cleaned.csv')
out_sample_df = pd.read_csv('/mnt/iusers01/msc-stu/hum-msc-data-sci-2024-2025/d14099as/scratch/thesis/out_sample_cleaned.csv')

# %%
in_sample_df = in_sample_df[['PERMNO', 'DlyCalDt', 'ExcessReturn']]
out_sample_df = out_sample_df[['PERMNO', 'DlyCalDt', 'ExcessReturn']]

print(in_sample_df.info())
print(out_sample_df.info())

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
y_test = pd.Series(targets)
results = pd.DataFrame(records)

# %%
def r2(y_true, y_pred):
    return 1-(((y_true-y_pred)**2).sum()/(y_true**2).sum())

def directional_accuracy(y_true, y_pred):
    sign_match = np.sign(y_true) == np.sign(y_pred)
    up_da = sign_match[y_true > 0].mean() if np.any(y_true > 0) else np.nan
    down_da = sign_match[y_true < 0].mean() if np.any(y_true < 0) else np.nan
    return up_da, down_da

# %%
# Chronos-Bolt-Tiny
pipeline = BaseChronosPipeline.from_pretrained('amazon/chronos-bolt-tiny', device_map=device_map, torch_dtype=torch.float32)
preds = [pipeline.predict_quantiles(context=[ctx.to(device)], prediction_length=1, quantile_levels=[0.5])[1].cpu().squeeze().item() for ctx in contexts]
y_chr_bolt_tiny = pd.Series(preds)
results['y_chr_bolt_tiny'] = y_chr_bolt_tiny
r2_chr_bolt_tiny = r2(y_test, y_chr_bolt_tiny)
mse_chr_bolt_tiny = mean_squared_error(y_test, y_chr_bolt_tiny)
mae_chr_bolt_tiny = mean_absolute_error(y_test, y_chr_bolt_tiny)
da_chr_bolt_tiny = (np.sign(y_test) == np.sign(y_chr_bolt_tiny)).mean()
up_da_chr_bolt_tiny, down_da_chr_bolt_tiny = directional_accuracy(y_test, y_chr_bolt_tiny)

# %%
# Chronos-Bolt-Mini
pipeline = BaseChronosPipeline.from_pretrained('amazon/chronos-bolt-mini', device_map=device_map, torch_dtype=torch.float32)
preds = [pipeline.predict_quantiles(context=[ctx.to(device)], prediction_length=1, quantile_levels=[0.5])[1].cpu().squeeze().item() for ctx in contexts]
y_chr_bolt_mini = pd.Series(preds)
results['y_chr_bolt_mini'] = y_chr_bolt_mini
r2_chr_bolt_mini = r2(y_test, y_chr_bolt_mini)
mse_chr_bolt_mini = mean_squared_error(y_test, y_chr_bolt_mini)
mae_chr_bolt_mini = mean_absolute_error(y_test, y_chr_bolt_mini)
da_chr_bolt_mini = (np.sign(y_test) == np.sign(y_chr_bolt_mini)).mean()
up_da_chr_bolt_mini, down_da_chr_bolt_mini = directional_accuracy(y_test, y_chr_bolt_mini)

# %%
# Chronos-Bolt-Small
pipeline = BaseChronosPipeline.from_pretrained('amazon/chronos-bolt-small', device_map=device_map, torch_dtype=torch.float32)
preds = [pipeline.predict_quantiles(context=[ctx.to(device)], prediction_length=1, quantile_levels=[0.5])[1].cpu().squeeze().item() for ctx in contexts]
y_chr_bolt_small = pd.Series(preds)
results['y_chr_bolt_small'] = y_chr_bolt_small
r2_chr_bolt_small = r2(y_test, y_chr_bolt_small)
mse_chr_bolt_small = mean_squared_error(y_test, y_chr_bolt_small)
mae_chr_bolt_small = mean_absolute_error(y_test, y_chr_bolt_small)
da_chr_bolt_small = (np.sign(y_test) == np.sign(y_chr_bolt_small)).mean()
up_da_chr_bolt_small, down_da_chr_bolt_small = directional_accuracy(y_test, y_chr_bolt_small)

# %%
# Chronos-Bolt-Base
pipeline = BaseChronosPipeline.from_pretrained('amazon/chronos-bolt-base', device_map=device_map, torch_dtype=torch.float32)
preds = [pipeline.predict_quantiles(context=[ctx.to(device)], prediction_length=1, quantile_levels=[0.5])[1].cpu().squeeze().item() for ctx in contexts]
y_chr_bolt_base = pd.Series(preds)
results['y_chr_bolt_base'] = y_chr_bolt_base
r2_chr_bolt_base = r2(y_test, y_chr_bolt_base)
mse_chr_bolt_base = mean_squared_error(y_test, y_chr_bolt_base)
mae_chr_bolt_base = mean_absolute_error(y_test, y_chr_bolt_base)
da_chr_bolt_base = (np.sign(y_test) == np.sign(y_chr_bolt_base)).mean()
up_da_chr_bolt_base, down_da_chr_bolt_base = directional_accuracy(y_test, y_chr_bolt_base)

# %%
# Switch to CPU for T5
device_map = "cpu"
device = torch.device("cpu")

# %%
contexts = []
targets = []
for id, grp in combined_df.groupby(ID_COL):
    values = grp[TARGET_COL].values
    dates = grp[DATE_COL].values

    for i in range(len(values) - WINDOW):
        pred_date = dates[i + WINDOW]
        if pred_date >= pd.to_datetime("2016-01-01"):
            contexts.append(torch.tensor(values[i:i+WINDOW], dtype=torch.float32, device=device))
            targets.append(values[i+WINDOW])

# %%
# Chronos-T5-Tiny
pipeline = BaseChronosPipeline.from_pretrained('amazon/chronos-t5-tiny', device_map=device_map, torch_dtype=torch.float32)
preds = [pipeline.predict_quantiles(context=[ctx.to(device)], prediction_length=1, quantile_levels=[0.5])[1].cpu().squeeze().item() for ctx in contexts]
y_chr_t5_tiny = pd.Series(preds)
results['y_chr_t5_tiny'] = y_chr_t5_tiny
r2_chr_t5_tiny = r2(y_test, y_chr_t5_tiny)
mse_chr_t5_tiny = mean_squared_error(y_test, y_chr_t5_tiny)
mae_chr_t5_tiny = mean_absolute_error(y_test, y_chr_t5_tiny)
da_chr_t5_tiny = (np.sign(y_test) == np.sign(y_chr_t5_tiny)).mean()
up_da_chr_t5_tiny, down_da_chr_t5_tiny = directional_accuracy(y_test, y_chr_t5_tiny)

# %%
# Chronos-T5-Mini
pipeline = BaseChronosPipeline.from_pretrained('amazon/chronos-t5-mini', device_map=device_map, torch_dtype=torch.float32)
preds = [pipeline.predict_quantiles(context=[ctx.to(device)], prediction_length=1, quantile_levels=[0.5])[1].cpu().squeeze().item() for ctx in contexts]
y_chr_t5_mini = pd.Series(preds)
results['y_chr_t5_mini'] = y_chr_t5_mini
r2_chr_t5_mini = r2(y_test, y_chr_t5_mini)
mse_chr_t5_mini = mean_squared_error(y_test, y_chr_t5_mini)
mae_chr_t5_mini = mean_absolute_error(y_test, y_chr_t5_mini)
da_chr_t5_mini = (np.sign(y_test) == np.sign(y_chr_t5_mini)).mean()
up_da_chr_t5_mini, down_da_chr_t5_mini = directional_accuracy(y_test, y_chr_t5_mini)

# %%
# Chronos-T5-Small
pipeline = BaseChronosPipeline.from_pretrained('amazon/chronos-t5-small', device_map=device_map, torch_dtype=torch.float32)
preds = [pipeline.predict_quantiles(context=[ctx.to(device)], prediction_length=1, quantile_levels=[0.5])[1].cpu().squeeze().item() for ctx in contexts]
y_chr_t5_small = pd.Series(preds)
results['y_chr_t5_small'] = y_chr_t5_small
r2_chr_t5_small = r2(y_test, y_chr_t5_small)
mse_chr_t5_small = mean_squared_error(y_test, y_chr_t5_small)
mae_chr_t5_small = mean_absolute_error(y_test, y_chr_t5_small)
da_chr_t5_small = (np.sign(y_test) == np.sign(y_chr_t5_small)).mean()
up_da_chr_t5_small, down_da_chr_t5_small = directional_accuracy(y_test, y_chr_t5_small)

# %%
# Final Results Table
results_matrix = [
    {"Model": "Chronos-Bolt-Tiny", "R-squared": r2_chr_bolt_tiny, "MSE": mse_chr_bolt_tiny, "MAE": mae_chr_bolt_tiny, "Direction Accuracy": da_chr_bolt_tiny, "Up Directional Accuracy": up_da_chr_bolt_tiny, "Down Directional Accuracy": down_da_chr_bolt_tiny},
    {"Model": "Chronos-Bolt-Mini", "R-squared": r2_chr_bolt_mini, "MSE": mse_chr_bolt_mini, "MAE": mae_chr_bolt_mini, "Direction Accuracy": da_chr_bolt_mini, "Up Directional Accuracy": up_da_chr_bolt_mini, "Down Directional Accuracy": down_da_chr_bolt_mini},
    {"Model": "Chronos-Bolt-Small", "R-squared": r2_chr_bolt_small, "MSE": mse_chr_bolt_small, "MAE": mae_chr_bolt_small, "Direction Accuracy": da_chr_bolt_small, "Up Directional Accuracy": up_da_chr_bolt_small, "Down Directional Accuracy": down_da_chr_bolt_small},
    {"Model": "Chronos-Bolt-Base", "R-squared": r2_chr_bolt_base, "MSE": mse_chr_bolt_base, "MAE": mae_chr_bolt_base, "Direction Accuracy": da_chr_bolt_base, "Up Directional Accuracy": up_da_chr_bolt_base, "Down Directional Accuracy": down_da_chr_bolt_base},
    {"Model": "Chronos-T5-Tiny", "R-squared": r2_chr_t5_tiny, "MSE": mse_chr_t5_tiny, "MAE": mae_chr_t5_tiny, "Direction Accuracy": da_chr_t5_tiny, "Up Directional Accuracy": up_da_chr_t5_tiny, "Down Directional Accuracy": down_da_chr_t5_tiny},
    {"Model": "Chronos-T5-Mini", "R-squared": r2_chr_t5_mini, "MSE": mse_chr_t5_mini, "MAE": mae_chr_t5_mini, "Direction Accuracy": da_chr_t5_mini, "Up Directional Accuracy": up_da_chr_t5_mini, "Down Directional Accuracy": down_da_chr_t5_mini},
    {"Model": "Chronos-T5-Small", "R-squared": r2_chr_t5_small, "MSE": mse_chr_t5_small, "MAE": mae_chr_t5_small, "Direction Accuracy": da_chr_t5_small, "Up Directional Accuracy": up_da_chr_t5_small, "Down Directional Accuracy": down_da_chr_t5_small}
]

results_matrix_df = pd.DataFrame(results_matrix)
results_matrix_df.to_csv("results_chronos_21_day.csv", index=False)

# %%
# Save full predictions
results.to_csv("chronos(21-day).csv", index=False)
