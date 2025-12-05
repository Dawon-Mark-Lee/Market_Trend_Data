# Market_Trend_Data
Market forecasting project using ML (Linear Regression, Random Forest) to predict market price and direction. Features include technical and macroeconomic factors. Backtesting showed the strategy outperformed Buy and Hold (421% vs 161%).

Market Trend Analysis and Trading Strategy BacktestThis project develops and tests a quantitative approach to predicting market movements using a combination of market data and external factors (sentiment, geopolitical risk, currency). It employs machine learning for forecasting and statistical analysis to assess feature impact. The final stage validates a directional trading strategy against a Buy and Hold benchmark.

⚙️ Methodology and Data
The analysis focuses on predicting next-day outcomes using daily time series data.

Data Preparation and Feature EngineeringThe model utilizes 12 features, incorporating market price data and derived technical and macroeconomic indicators.Derived Features: Simple Moving Averages (SMA_10, SMA_50), 20-day Volatility ($Volatility\_20D$).

External Features: Lagged values (previous day's score) were used for VIX, Sentiment, Geopolitical Risk, and Currency Index. These variables are critical for predicting market direction.

Model Scenarios
The project implemented three distinct modeling scenarios:
Price Forecasting (Regression): Predicts the exact next-day closing price (Target_Price) using Linear Regression.
Direction Classification: Predicts if the market will close "Up" or "Down" (Target_Up) using a Random Forest Classifier.
Volatility Forecasting (Regression): Predicts the next-day VIX value (Target_VIX).

📈 Model Performance and Feature Insights
1. Price Forecasting Performance (Scenario 1)
The Linear Regression model demonstrated extremely high accuracy in predicting the next day's price.
Model: Linear RegressionR-squared (R^2): 0.9996
Conclusion: The model is an excellent fit for tracking the next day's price, suggesting high linearity in short-term price movements.

2. Feature Importance for Directional Prediction
The features driving the Up/Down Classification model were examined. External, lagged indicators held the highest influence.
Rank Feature Importance Score
1Lag_1_GeoPolitical_Risk_Score approx 0.103$
2Lag_1_Currency_Index approx 0.102
3Lag_1_Sentiment_Score approx 0.101
4Lag_1_VIX_Close approx 0.101
5Daily_Return_Pct approx 0.098
Conclusion: The model relies most heavily on macro-level and volatility indicators from the previous day to determine the direction of the next day's movement.

4. Classification Performance (Scenario 2)
The Random Forest Classifier used for directional prediction showed limited predictive power.
Model: Random Forest Classifier
AUC Score: 0.5186
Conclusion: The AUC score is only slightly above 0.50, indicating the model's ability to distinguish between up and down days is weak. This suggests further feature engineering or advanced models are needed.

🔬 Macroeconomic Impact Analysis
Statistical tests assessed the relationship between external factors and subsequent market returns.
Policy Decision Impact (T-test)
The analysis compared next-day returns after a Federal Rate Change Flag was raised versus no change.
Mean Return (After Policy Change): 0.0264%
$Mean Return (After No Change): 0.0225%
P-value: $0.9317$
Conclusion: With a P-value above 0.05, there is no statistically significant difference in mean returns following a policy change.

Market-Sentiment Relationship (OLS Regression)
An Ordinary Least Squares (OLS) regression modeled the next-day return as a function of the previous day's sentiment score.
Sentiment Coefficient: -0.0004
Conclusion: The negative coefficient indicates a very small, inverse relationship between the previous day's sentiment and the next day's market return.

💰 Backtesting Results
A simple trading strategy was implemented on the test data: Buy and hold when the classification model predicts an "Up" day (1), and stay in cash (0% return) otherwise.
Cumulative Returns (Test Period:1997-2017)
Strategy
Total Cumulative Return
Model Strategy 421.13%
Buy and Hold 160.83%
ConclusionDespite the low AUC score of the classification model, the trading strategy significantly outperformed the Buy and Hold benchmark over the test period. This suggests that even marginally predictive signals, when used to avoid major downturns (seen around 2008 and 2012 in the Buy and Hold line), can generate substantial alpha. The majority of the outperformance occurred in the latter half of the test period.
