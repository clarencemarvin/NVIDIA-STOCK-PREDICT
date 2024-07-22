# NVIDIA Stock Price Prediction using XGBoost, LSTM, MAMM, and Ensemble Methods

This project demonstrates the implementation of stock price prediction using XGBoost, LSTM, and ensemble methods. It combines the predictions from individual models to generate more accurate forecasts. The project also incorporates the Tiingo API for fetching historical stock data.

## Requirements

To run this project, you need the following dependencies:

- Python (version 3.6 or higher)
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- TensorFlow
- Keras
- Tiingo

## Usage

1. Obtain a Tiingo API token by signing up on the [Tiingo website](https://www.tiingo.com/).

2. Set the `TIINGO_API_KEY` environment variable with your Tiingo API token. You can do this by running the following command in your terminal or adding it to your environment variables: export TIINGO_API_KEY="your_api_token_here"

3. Update the `ticker` variable in the code with the desired stock ticker symbol you want to predict.

4. Run the script using the following command: python predictor.py

The script will fetch the historical stock data using the Tiingo API, preprocess the data, train the XGBoost and LSTM models, and generate predictions using individual models and ensemble methods.

5. The script will output the following metrics for each model and ensemble method:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (R2) score

It will also display the accuracy scores for the test set and training set.

## Explanation

The project follows these main steps:

1. **Data Retrieval**: The script fetches historical stock data using the Tiingo API. The Tiingo API provides a reliable and efficient way to access financial data. It is used as an alternative to the Yahoo Finance API (yfinance) because yfinance can sometimes be buggy or inconsistent.

2. **Data Preprocessing**: The retrieved stock data is preprocessed by removing missing values, scaling the features using MinMaxScaler, and creating input sequences and corresponding labels for training the models.

3. **Model Training**: The script trains two models:
- XGBoost: An optimized gradient boosting algorithm that combines weak learners to create a strong predictive model.
- LSTM (Long Short-Term Memory): A type of recurrent neural network capable of learning long-term dependencies in sequential data.

4. **Ensemble Methods**: The predictions from the XGBoost and LSTM models are combined using a weighted ensemble approach. The weights are determined based on the performance of each model. Additionally, a combined ensemble method is used, which incorporates moving average techniques to further improve the predictions.

5. **Evaluation**: The script evaluates the performance of each model and ensemble method using various metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) score. It also calculates the accuracy scores for the test set and training set.

By leveraging the strengths of different models and ensemble techniques, this project aims to provide more accurate and reliable stock price predictions. The use of the Tiingo API ensures a stable and consistent data source for training and testing the models.


## Result
XGBoost - MSE: 1290.67, MAE: 24.18, R2: -0.76
LSTM - MSE: 1101.13, MAE: 22.62, R2: -0.50
Weighted Ensemble - MSE: 1191.67, MAE: 23.16, R2: -0.62
Combined Ensemble and MA - MSE: 600.58, MAE: 16.41, R2: 0.18
Test Set Accuracy: 0.7434
Training Set Accuracy: 0.6600

<img width="1440" alt="Screenshot 2024-07-22 at 8 38 57 PM" src="https://github.com/user-attachments/assets/665defe4-1ba2-48c0-8c29-c8ba02ac4a6e">
