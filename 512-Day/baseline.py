# %%
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, TweedieRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# %%
in_sample_df = pd.read_csv('/mnt/iusers01/msc-stu/hum-msc-data-sci-2024-2025/d14099as/scratch/thesis/in_sample_cleaned.csv')
out_sample_df = pd.read_csv('/mnt/iusers01/msc-stu/hum-msc-data-sci-2024-2025/d14099as/scratch/thesis/out_sample_cleaned.csv')

# %%
# Creating Previous n-Day Returns
estimation_window = 512

def create_previous_day_returns(permno, in_sample_df, out_sample_df, estimation_window):
    # Iterating over each unique stock (PERMNO) in the in-sample data
    stock_in = in_sample_df[in_sample_df['PERMNO'] == permno].sort_values('DlyCalDt')
    stock_out = out_sample_df[out_sample_df['PERMNO'] == permno].sort_values('DlyCalDt')

    combined_df = pd.concat([stock_in, stock_out]).sort_values('DlyCalDt').reset_index(drop=True)

    # Create lag features
    for lag in range(1, estimation_window + 1):
        combined_df[f'{lag}DayRet'] = combined_df['ExcessReturn'].shift(lag)
    
    return combined_df

total_stock_df = pd.DataFrame()

for permno in in_sample_df['PERMNO'].unique():
    combined_df = create_previous_day_returns(permno, in_sample_df, out_sample_df, estimation_window)
    
    # Append the results to a master DataFrame
    total_stock_df = pd.concat([total_stock_df,combined_df], ignore_index=True)
    
total_stock_df.sort_values(['DlyCalDt','Ticker'], inplace=True)
total_stock_df.dropna(inplace=True)
total_stock_df['DlyCalDt'] = pd.to_datetime(total_stock_df['DlyCalDt'])

# %%
total_stock_df

# %% [markdown]
# ### Approach 2

# %% [markdown]
# The model is trained for all 50 stocks, and the results are calculated. The model is retrained every n days (tunable) by expanding the training dates

# %%
out_sample_start = pd.to_datetime('2016-01-01')
alpha = 0.001
estimation_window = 512  # set here, avoid overriding inside function

def rolling_window(total_stock_df, input_model, estimation_window=estimation_window, alpha=alpha):
    y_preds = []
    y_actuals = []
    predictions = []  # Collect predictions with metadata


    # Train once on all data before out_sample_start
    train_data = total_stock_df[total_stock_df['DlyCalDt'] < out_sample_start]

    # One-hot encode tickers in train
    ticker_dummies_train = pd.get_dummies(train_data['PERMNO'], prefix='PERMNO')

    # Scale lagged features in train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data[[f'{lag}DayRet' for lag in range(1, estimation_window + 1)]])
    X_train = np.concatenate([X_train_scaled, ticker_dummies_train.values], axis=1)
    y_train = train_data['ExcessReturn']

    # Initialize model
    if input_model == 'Ridge':
        model = Ridge(alpha=alpha)
    elif input_model == 'OLS':
        model = LinearRegression()
    elif input_model == 'ElasticNet':
        model = ElasticNet(alpha=alpha)
    elif input_model == 'Lasso':
        model = Lasso(alpha=alpha)
    elif input_model == 'GLM':
        model = TweedieRegressor(power=0, alpha=alpha)
    elif input_model == 'RF':
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=10, max_depth=5, max_features=0.5, n_jobs=-1)
    elif input_model == 'GBRT':
        model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, loss='huber')
    elif input_model == 'NN1':
        model = MLPRegressor(hidden_layer_sizes=(64,), max_iter=1000, activation='relu', solver='adam', learning_rate='adaptive', early_stopping=True)
    elif input_model == 'NN2':
        model = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=1000, activation='relu', solver='adam', learning_rate='adaptive', early_stopping=True)
    elif input_model == 'NN3':
        model = MLPRegressor(hidden_layer_sizes=(128,128,64), max_iter=1000, activation='relu', solver='adam', learning_rate='adaptive', early_stopping=True)
    elif input_model == 'NN4':
        model = MLPRegressor(hidden_layer_sizes=(256,128,64,32), max_iter=1000, activation='relu', solver='adam', learning_rate='adaptive', early_stopping=True)
    elif input_model == 'NN5':
        model = MLPRegressor(hidden_layer_sizes=(256,128,64,32,16), max_iter=1000, activation='relu', solver='adam', learning_rate='adaptive', early_stopping=True)

    model.fit(X_train, y_train)

    # Predict for each date >= out_sample_start
    unique_dates = total_stock_df['DlyCalDt'].unique()
    for current_date in unique_dates:
        if current_date < out_sample_start:
            continue

        test_data = total_stock_df[total_stock_df['DlyCalDt'] == current_date]
        if test_data.empty:
            continue

        ticker_dummies_test = pd.get_dummies(test_data['PERMNO'], prefix='PERMNO')
        # Align test dummies with train dummies
        for col in ticker_dummies_train.columns:
            if col not in ticker_dummies_test.columns:
                ticker_dummies_test[col] = 0
        ticker_dummies_test = ticker_dummies_test[ticker_dummies_train.columns]

        X_test_scaled = scaler.transform(test_data[[f'{lag}DayRet' for lag in range(1, estimation_window + 1)]])
        X_test = np.concatenate([X_test_scaled, ticker_dummies_test.values], axis=1)

        y_test = test_data['ExcessReturn']

        y_pred = model.predict(X_test)
        # Save predictions with metadata
        predictions.extend([
            {'Date': current_date, 'PERMNO': permno, 'Actual': actual, 'Predicted': pred}
            for permno, actual, pred in zip(test_data['PERMNO'], y_test.values, y_pred)
        ])
        y_preds.extend(y_pred)
        y_actuals.extend(y_test.values)

    y_actuals_np = np.array(y_actuals)
    y_preds_np = np.array(y_preds)
    sign_match = np.sign(y_actuals_np) == np.sign(y_preds_np)

    # Up and Down masks
    up_da = sign_match[y_actuals_np > 0].mean() if np.any(y_actuals_np > 0) else np.nan
    down_da = sign_match[y_actuals_np < 0].mean() if np.any(y_actuals_np < 0) else np.nan
    mse = mean_squared_error(y_actuals, y_preds)
    r2 = r2_score(y_actuals, y_preds)
    mae = mean_absolute_error(y_actuals, y_preds)
    da = (np.sign(y_actuals) == np.sign(y_preds)).mean()

    return mse, r2, mae, da, up_da, down_da, predictions


# %%
# %%
# Implementing the rolling window for different models
models = ['Ridge', 'OLS', 'ElasticNet', 'Lasso', 'GLM', 'RF', 'NN1', 'NN2', 'NN3', 'NN4', 'NN5', 'GBRT']
results = []
wide_preds_df = pd.DataFrame()
estimation_window = 512

for alpha in [0.001]:
    print(f"Running models with alpha: {alpha}")
    for model in models:

        mse, r2, mae, da, up_da, down_da, predictions = rolling_window(total_stock_df, input_model=model, estimation_window=estimation_window, alpha=alpha)
        print(f"Model: {model}, MSE: {mse:.7f}, R2: {r2:.7f}, MAE: {mae:.7f}, Direction Accuracy: {da:.7f}, Up Directional Accuracy: {up_da:.7f}, Down Directional Accuracy: {down_da:.7f}")

        results.append({
            'Model': model,
            'MSE': mse,
            'R2': r2,
            'MAE': mae,
            'Direction Accuracy': da,
            'Up Directional Accuracy': up_da,
            'Down Directional Accuracy': down_da
        })

        # Convert predictions list to DataFrame
        pred_df = pd.DataFrame(predictions)
        pred_df.rename(
            columns={'Date': 'DlyCalDt', 'Actual': 'ExcessReturn', 'Predicted': f'{model}'},
            inplace=True
        )

        # Merge predictions into wide_preds_df
        if wide_preds_df.empty:
            wide_preds_df = pred_df[['PERMNO', 'DlyCalDt', 'ExcessReturn', f'{model}']]
        else:
            wide_preds_df = wide_preds_df.merge(
                pred_df[['PERMNO', 'DlyCalDt', f'{model}']],
                on=['PERMNO', 'DlyCalDt'],
                how='left'
            )


# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df.sort_values(by='Model', inplace=True)
results_df.to_csv('model_results.csv (512-window).csv', index=False)
# Save predictions to CSV
wide_preds_df.sort_values(['DlyCalDt', 'PERMNO'], inplace=True)
wide_preds_df.to_csv('model_predictions.csv (512-window).csv', index=False)
