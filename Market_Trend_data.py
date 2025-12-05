import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, roc_auc_score
from scipy import stats
import statsmodels.api as sm

# Load the full prepared dataset
df = pd.read_csv("Market_Trend_External.csv", index_col='Date', parse_dates=True)

# Create missing features
df['SMA_10'] = df['Close_Price'].rolling(window=10).mean()
df['SMA_50'] = df['Close_Price'].rolling(window=50).mean()
df['Volatility_20D'] = df['Daily_Return_Pct'].rolling(window=20).std()
df['Lag_1_VIX_Close'] = df['VIX_Close'].shift(1)
df['Lag_1_Sentiment_Score'] = df['Sentiment_Score'].shift(1)
df['Lag_1_GeoPolitical_Risk_Score'] = df['GeoPolitical_Risk_Score'].shift(1)
df['Lag_1_Currency_Index'] = df['Currency_Index'].shift(1)

# Define the final feature list
features = [
    'Open_Price', 'Daily_Return_Pct', 'Volatility_Range', 'Economic_News_Flag',
    'Federal_Rate_Change_Flag', 'SMA_10', 'SMA_50', 'Volatility_20D',
    'Lag_1_VIX_Close', 'Lag_1_Sentiment_Score', 'Lag_1_GeoPolitical_Risk_Score',
    'Lag_1_Currency_Index'
]

# Drop rows with NaN values created by rolling/shifting operations
df = df.dropna()

# Create target variables
df['Target_Price'] = df['Close_Price'].shift(-1)  # Next day closing price
df['Target_Up'] = (df['Close_Price'].shift(-1) > df['Close_Price']).astype(int)  # 1 if price goes up, 0 otherwise
df['Target_VIX'] = df['VIX_Close'].shift(-1)  # Next day VIX

# Drop the last row since it won't have a target
df = df.iloc[:-1]

# Define split point (70% for training)
split_point = int(len(df) * 0.7)

# Split features (X) and target (y)
X = df[features]
y_price = df['Target_Price']
y_up = df['Target_Up']
y_vix = df['Target_VIX']
y_return = df['Daily_Return_Pct'].shift(-1).dropna() # Next day return for statistical analysis

# Train/Test Split
X_train, X_test = X[:split_point], X[split_point:]
y_train_price, y_test_price = y_price[:split_point], y_price[split_point:]
y_train_up, y_test_up = y_up[:split_point], y_up[split_point:]
y_train_vix, y_test_vix = y_vix[:split_point], y_vix[split_point:]

# Align y_return for T-test and OLS, dropping last row for consistent length
df_test = df[split_point:].copy().iloc[:-1] # Test data for backtesting/reporting

# --------------------------------------------------------------------------

# --- Scenario A: Price Forecasting (Regression) ---
lr_price = LinearRegression()
lr_price.fit(X_train, y_train_price)
y_pred_price = lr_price.predict(X_test)

rmse_price = np.sqrt(mean_squared_error(y_test_price, y_pred_price))
r2_price = r2_score(y_test_price, y_pred_price)

print("--- Scenario A: Price Forecasting ---")
print(f"Model: Linear Regression")
print(f"RMSE (Price Error): {rmse_price:.4f}")
print(f"R-squared: {r2_price:.4f}\n")

# --- Scenario B: Up/Down Day Classification ---
# Using class_weight='balanced' to handle potential imbalance in up/down days
rf_up = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_up.fit(X_train, y_train_up)
y_pred_up = rf_up.predict(X_test)
y_prob_up = rf_up.predict_proba(X_test)[:, 1]

auc_up = roc_auc_score(y_test_up, y_prob_up)

print("--- Scenario B: Up/Down Classification ---")
print("Model: Random Forest Classifier")
print(f"AUC Score: {auc_up:.4f}")
print("Classification Report:")
print(classification_report(y_test_up, y_pred_up, zero_division=0))

# --- Scenario C: Volatility Prediction ---
lr_vix = LinearRegression()
lr_vix.fit(X_train, y_train_vix)
y_pred_vix = lr_vix.predict(X_test)

rmse_vix = np.sqrt(mean_squared_error(y_test_vix, y_pred_vix))
r2_vix = r2_score(y_test_vix, y_pred_vix)

print("--- Scenario C: Volatility Prediction ---")
print(f"Model: Linear Regression")
print(f"RMSE (VIX Error): {rmse_vix:.4f}")
print(f"R-squared: {r2_vix:.4f}\n")

# Save predictions for backtesting (Phase 3)
df_backtest = df[split_point:].copy()
df_backtest['Prediction_Up'] = rf_up.predict(X_test)
df_backtest['Prediction_Price'] = y_pred_price
df_backtest['Actual_Price'] = y_test_price.values
df_backtest['Actual_Up'] = y_test_up.values
df_backtest.to_csv("Test_Data_with_Predictions.csv")
print("Backtesting results saved to Test_Data_with_Predictions.csv\n")

# --------------------------------------------------------------------------

# Load data, calculate next-day return for analysis
df = pd.read_csv("Market_Trend_External.csv", index_col='Date', parse_dates=True)
df['Next_Day_Return'] = df['Close_Price'].pct_change().shift(-1) * 100
df.dropna(subset=['Next_Day_Return'], inplace=True)

# Align data to features used (lagged flags)
df['Policy_Change_Lag'] = df['Federal_Rate_Change_Flag'].shift(1)
df['Lag_1_Sentiment_Score'] = df['Sentiment_Score'].shift(1)
df.dropna(subset=['Policy_Change_Lag', 'Lag_1_Sentiment_Score'], inplace=True)

# Use the full dataset for impact analysis to maximize sample size
df_impact = df.copy()

print("--- Scenario D: Macroeconomic and Policy Impact Analysis ---")

# --- Policy Decision Impact (T-test) ---
# Group returns based on whether a federal rate change occurred the previous day
returns_with_change = df_impact[df_impact['Policy_Change_Lag'] == 1]['Next_Day_Return']
returns_no_change = df_impact[df_impact['Policy_Change_Lag'] == 0]['Next_Day_Return']

# Perform independent two-sample T-test
t_stat, p_value = stats.ttest_ind(returns_with_change, returns_no_change, equal_var=False)

print("\n**Policy Decision Impact (Federal Rate Change)**")
print(f"Mean Return (After Policy Change): {returns_with_change.mean():.4f}%")
print(f"Mean Return (After No Change): {returns_no_change.mean():.4f}%")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print("Insight: A small P-value (e.g., less than 0.05) rejects the idea that the means are equal.\n")

# --- Market-Sentiment Relationship Modeling (OLS Regression) ---
# Model: Next_Day_Return is explained by Lag_1_Sentiment_Score
X_ols = df_impact['Lag_1_Sentiment_Score']
Y_ols = df_impact['Next_Day_Return']
X_ols = sm.add_constant(X_ols) # Add constant for intercept

ols_model = sm.OLS(Y_ols, X_ols).fit()

print("**Market-Sentiment Relationship Modeling (OLS)**")
print(ols_model.summary().tables[1]) # Print the coefficients table
print("\nInsight: The coefficient for Lag_1_Sentiment_Score shows the average percentage point change in next day's return for a one-unit change in the sentiment score.")

# --------------------------------------------------------------------------

# --- Phase 3: Backtesting ---
# Load the saved predictions
df_backtest_results = pd.read_csv("Test_Data_with_Predictions.csv", index_col='Date', parse_dates=True)

# Calculate the actual return realized on the next day (needed for backtesting)
df_backtest_results['Actual_Return_Pct'] = df_backtest_results['Close_Price'].pct_change().shift(-1) * 100
df_backtest_results = df_backtest_results.dropna(subset=['Actual_Return_Pct'])

# Backtesting Logic:
# Strategy Return = Actual Return if model predicted UP (1), otherwise 0 (hold cash)
df_backtest_results['Strategy_Return'] = df_backtest_results.apply(
    lambda row: row['Actual_Return_Pct'] if row['Prediction_Up'] == 1 else 0,
    axis=1
)

# Comparison: Buy and Hold Return
df_backtest_results['Buy_Hold_Return'] = df_backtest_results['Actual_Return_Pct']

# Calculate Cumulative Returns (starting at 1)
df_backtest_results['Strategy_Cumulative'] = (1 + df_backtest_results['Strategy_Return'] / 100).cumprod()
df_backtest_results['Buy_Hold_Cumulative'] = (1 + df_backtest_results['Buy_Hold_Return'] / 100).cumprod()

# --- Plotting ---
plt.figure(figsize=(12, 6))
plt.plot(df_backtest_results.index, df_backtest_results['Strategy_Cumulative'], label='Model Strategy')
plt.plot(df_backtest_results.index, df_backtest_results['Buy_Hold_Cumulative'], label='Buy and Hold')
plt.title('Backtesting Cumulative Returns (Test Period)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (Starting at 1)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('backtesting_cumulative_returns_final.png')
plt.close()

# Report final metrics
strategy_final_return = (df_backtest_results['Strategy_Cumulative'].iloc[-1] - 1) * 100
buy_hold_final_return = (df_backtest_results['Buy_Hold_Cumulative'].iloc[-1] - 1) * 100

print("\n--- Phase 3: Backtesting Results ---")
print(f"Model Strategy Total Return: {strategy_final_return:.2f}%")
print(f"Buy and Hold Total Return: {buy_hold_final_return:.2f}%")

# Save the backtesting results
df_backtest_results[['Strategy_Cumulative', 'Buy_Hold_Cumulative', 'Strategy_Return', 'Buy_Hold_Return']].to_csv('final_backtesting_results.csv')

# ---------------------------------------------------------------------------

# Load the test data and the trained Random Forest model is still needed

# --- Extract and Plot Feature Importances ---
importances = rf_up.feature_importances_
feature_names = X_train.columns
feature_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_series.plot(kind='barh', color='skyblue')
plt.title('Feature Importance for Up/Down Classification Model')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Save the feature importances to CSV
feature_series.to_csv('feature_importances_final.csv', header=['Importance'])

print("--- Phase 4: Final Reporting and Feature Importance ---")
print("\nTop 5 Feature Importances for Classification:")
print(feature_series.head())

# ---------------------------------------------------------------------------

# ... (Code to load data, define features, and split the data)

# --- Scenario A: Price Forecasting (Regression) ---
lr_price = LinearRegression()
lr_price.fit(X_train, y_train_price)
y_pred_price = lr_price.predict(X_test)

# Create DataFrame for plotting
results_df = pd.DataFrame({
    'Actual Price': y_test_price,
    'Predicted Price': y_pred_price
})

# --- Visualization: Actual vs. Predicted Price ---
plt.figure(figsize=(8, 8))
plt.scatter(results_df['Actual Price'], results_df['Predicted Price'], alpha=0.5, s=10)
plt.plot([results_df['Actual Price'].min(), results_df['Actual Price'].max()],
         [results_df['Actual Price'].min(), results_df['Actual Price'].max()],
         'r--', lw=2, label='Perfect Prediction Line')
plt.title('Actual vs. Predicted Next Day Price (Test Set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('actual_vs_predicted_price.png') # This line saves the file.
plt.close()