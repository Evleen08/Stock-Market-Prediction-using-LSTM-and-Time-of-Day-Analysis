# Stock Market Prediction using LSTM with Time-of-Day Analysis

Prediction of intraday stock price movements using a Long Short-Term Memory (LSTM) neural network enhanced with time-of-day features. The model is trained and evaluated on historical Tata Motors stock data.

## Overview

Stock market prediction is difficult due to the volatility and nonlinearity of price movements. While LSTM networks are well suited to modeling temporal dependencies, they typically don't account for time-of-day effects — the fact that trading behavior differs between market open, midday, and close.

This project extends a standard LSTM model with **time-of-day features** to better capture intraday patterns, with the goal of improving short-term prediction accuracy over a baseline LSTM.

## Key Idea

- Segment trading data by hour and session (forenoon / afternoon / peak hours)
- Encode time cyclically (sine/cosine) so the model understands that hours wrap around, rather than treating 11 PM and 12 AM as distant values
- Select the most relevant features with LASSO regression to reduce noise and overfitting
- Feed the resulting features into a multi-layer LSTM to forecast future prices

## System Architecture

1. **Data Collection** — historical OHLCV (open, high, low, close, volume) data for Tata Motors, sourced from Yahoo Finance
2. **Data Preprocessing** — linear interpolation for missing values, min-max normalization
3. **Time-of-Day Analysis** — extraction of hour/session features, cyclic time encoding
4. **Feature Selection** — LASSO regression to identify the most predictive features
5. **LSTM Model** — two-layer LSTM with dropout regularization, trained on the selected features
6. **Evaluation** — RMSE and MAE, benchmarked against a baseline LSTM without time-of-day features

## Methodology

- **Data split:** 80% train / 10% validation / 10% test
- **Training:** early stopping and model checkpoints to avoid overfitting
- **Hyperparameter tuning:** grid search with cross-validation over learning rate, number of LSTM units, and sequence length
- **Feature selection:** LASSO regression tested at α = 0.01, 0.05, and 0.1

## Results

The time-of-day-augmented LSTM outperformed the baseline LSTM on both RMSE and MAE, supporting the hypothesis that intraday temporal cues (peak hours, session-level patterns, cyclic encoding) improve short-term stock price forecasting. Full results, plots, and metric tables are in the project notebook.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebook/
│   └── stock_market_prediction.ipynb   # main analysis and model notebook
├── data/                                # dataset (see Dataset section below)
└── LICENSE
```

Adjust the structure above to match your actual file layout once everything is pushed.

## Dataset

This project uses historical Tata Motors stock data (OHLCV) from Yahoo Finance.

- If the dataset is small, it can be committed directly under `data/`.
- If it's large or you'd rather not version it, host it externally (e.g. Google Drive, Kaggle) and download it with the [`yfinance`](https://pypi.org/project/yfinance/) package or a direct link — document the exact steps here once decided.

## Getting Started

```bash
# clone the repo
git clone <your-repo-url>
cd <your-repo-name>

# install dependencies
pip install -r requirements.txt

# run the notebook
jupyter notebook notebook/stock_market_prediction.ipynb
```

## Tech Stack

- Python
- Pandas / NumPy — data handling
- scikit-learn — LASSO feature selection
- TensorFlow / Keras — LSTM model
- Matplotlib / Seaborn — visualization


## License

Add a license (e.g. MIT) if you intend for others to reuse this work — see [choosealicense.com](https://choosealicense.com/) for guidance.
