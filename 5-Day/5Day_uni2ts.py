# %%
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# %%
# Parameters and settings

WINDOW = 5
DATE_COL = 'DlyCalDt'
ID_COL = 'PERMNO'
TARGET_COL = 'ExcessReturn'

in_sample_start_date = pd.to_datetime("2000-01-01")
in_sample_end_date = pd.to_datetime("2015-12-31")
out_sample_start_date = pd.to_datetime("2016-01-01")
out_sample_end_date = pd.to_datetime("2024-12-31")

device_map = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_map)

# %%
# Load data

in_sample_df = pd.read_csv('/mnt/iusers01/msc-stu/hum-msc-data-sci-2024-2025/d14099as/scratch/thesis/in_sample_cleaned.csv')
out_sample_df = pd.read_csv('/mnt/iusers01/msc-stu/hum-msc-data-sci-2024-2025/d14099as/scratch/thesis/out_sample_cleaned.csv')


in_sample_df[DATE_COL] = pd.to_datetime(in_sample_df[DATE_COL])
out_sample_df[DATE_COL] = pd.to_datetime(out_sample_df[DATE_COL])

in_sample_df = in_sample_df[[ID_COL, DATE_COL, TARGET_COL]]
out_sample_df = out_sample_df[[ID_COL, DATE_COL, TARGET_COL]]

# %%
stocks_permno = in_sample_df["PERMNO"].unique().tolist()
print(f"Number of unique stocks: {len(stocks_permno)}")

# %%
combined_df = pd.concat([in_sample_df, out_sample_df])
combined_df = combined_df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)
combined_df[DATE_COL] = pd.to_datetime(combined_df[DATE_COL])

for lag in range(1, WINDOW+1):
    combined_df[f'lag_{lag}'] = combined_df.groupby(ID_COL)[TARGET_COL].shift(lag)
combined_df = combined_df.dropna(subset=[f'lag_{lag}' for lag in range(1, WINDOW+1)]).reset_index(drop=True)
combined_df.sort_values([ID_COL, DATE_COL], inplace=True)
combined_df.reset_index(drop=True, inplace=True)

# %%
records = []
for _, row in combined_df.iterrows():
    if row[DATE_COL] >= out_sample_start_date:
        context = [row[f'lag_{i}'] for i in range(WINDOW, 0, -1)]
        start_ts = row[DATE_COL] - pd.Timedelta(days=WINDOW)
        records.append({
            "start":  start_ts,
            "target": context
        })

test_ds = ListDataset(records, freq="D")

y_test = pd.Series(out_sample_df[TARGET_COL].values)
results = out_sample_df[[ID_COL, DATE_COL, TARGET_COL]]

# %%
def r2(y_true, y_pred):
    return 1 - (((y_true - y_pred) ** 2).sum() / (y_true ** 2).sum())

# %%
# Uni2ts-Moirai Small
moirai_s = MoiraiForecast(
    module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-small"),
    prediction_length=1,
    context_length=WINDOW,
    patch_size="auto",
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0
)
predictor = moirai_s.create_predictor(batch_size=32)
predictor.to(device)

preds = [forecast.mean[0] for forecast in predictor.predict(test_ds)]
y_moirai_s = pd.Series(preds)
results['y_moirai_s'] = y_moirai_s

sign_match = np.sign(y_test) == np.sign(y_moirai_s)
r2_moirai_s = r2(y_test, y_moirai_s)
mse_moirai_s = mean_squared_error(y_test, y_moirai_s)
mae_moirai_s = mean_absolute_error(y_test, y_moirai_s)
da_moirai_s = sign_match.mean()
up_da_moirai_s = sign_match[y_test > 0].mean() if np.any(y_test > 0) else np.nan
down_da_moirai_s = sign_match[y_test < 0].mean() if np.any(y_test < 0) else np.nan

# %%
# Uni2ts-Moirai-MoE Small
moirai_moe_s = MoiraiMoEForecast(
    module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-small"),
    prediction_length=1,
    context_length=WINDOW,
    patch_size=16,
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0
)
predictor = moirai_moe_s.create_predictor(batch_size=32)
predictor.to(device)

preds = [forecast.mean[0] for forecast in predictor.predict(test_ds)]
y_moirai_moe_s = pd.Series(preds)
results['y_moirai_moe_s'] = y_moirai_moe_s

sign_match = np.sign(y_test) == np.sign(y_moirai_moe_s)
r2_moirai_moe_s = r2(y_test, y_moirai_moe_s)
mse_moirai_moe_s = mean_squared_error(y_test, y_moirai_moe_s)
mae_moirai_moe_s = mean_absolute_error(y_test, y_moirai_moe_s)
da_moirai_moe_s = sign_match.mean()
up_da_moirai_moe_s = sign_match[y_test > 0].mean() if np.any(y_test > 0) else np.nan
down_da_moirai_moe_s = sign_match[y_test < 0].mean() if np.any(y_test < 0) else np.nan

# %%
# Uni2ts-Moirai-MoE Base
moirai_moe_b = MoiraiMoEForecast(
    module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-base"),
    prediction_length=1,
    context_length=WINDOW,
    patch_size=16,
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0
)
predictor = moirai_moe_b.create_predictor(batch_size=32)
predictor.to(device)

preds = [forecast.mean[0] for forecast in predictor.predict(test_ds)]
y_moirai_moe_b = pd.Series(preds)
results['y_moirai_moe_b'] = y_moirai_moe_b

sign_match = np.sign(y_test) == np.sign(y_moirai_moe_b)
r2_moirai_moe_b = r2(y_test, y_moirai_moe_b)
mse_moirai_moe_b = mean_squared_error(y_test, y_moirai_moe_b)
mae_moirai_moe_b = mean_absolute_error(y_test, y_moirai_moe_b)
da_moirai_moe_b = sign_match.mean()
up_da_moirai_moe_b = sign_match[y_test > 0].mean() if np.any(y_test > 0) else np.nan
down_da_moirai_moe_b = sign_match[y_test < 0].mean() if np.any(y_test < 0) else np.nan

# %%
# Compile results
results_matrix = [
    {
        "Model": "Uni2ts-Moirai Small",
        "R-squared": r2_moirai_s,
        "MSE": mse_moirai_s,
        "MAE": mae_moirai_s,
        "Direction Accuracy": da_moirai_s,
        "Up Directional Accuracy": up_da_moirai_s,
        "Down Directional Accuracy": down_da_moirai_s
    },
    {
        "Model": "Uni2ts-Moirai-MoE Small",
        "R-squared": r2_moirai_moe_s,
        "MSE": mse_moirai_moe_s,
        "MAE": mae_moirai_moe_s,
        "Direction Accuracy": da_moirai_moe_s,
        "Up Directional Accuracy": up_da_moirai_moe_s,
        "Down Directional Accuracy": down_da_moirai_moe_s
    },
    {
        "Model": "Uni2ts-Moirai-MoE Base",
        "R-squared": r2_moirai_moe_b,
        "MSE": mse_moirai_moe_b,
        "MAE": mae_moirai_moe_b,
        "Direction Accuracy": da_moirai_moe_b,
        "Up Directional Accuracy": up_da_moirai_moe_b,
        "Down Directional Accuracy": down_da_moirai_moe_b
    }
]

results_matrix_df = pd.DataFrame(results_matrix)
print(results_matrix_df)
results_matrix_df.to_csv("uni2ts(5-day)results.csv", index=False)

# %%
# Save predictions
results.to_csv("uni2ts(5-day)predictions.csv", index=False)

