📊 Sales Forecasting using Time Series Analysis

This project implements an end-to-end time series sales forecasting system using historical Walmart sales data.
It focuses on seasonality analysis, baseline forecasting, feature engineering, model evaluation, and a command-line forecasting interface (CLI).

The goal is to accurately forecast future weekly sales while following industry-standard time series practices.

🚀 Project Highlights

✔ Time-aware exploratory data analysis (EDA)

✔ Multiple baseline models (Naïve, Seasonal Naïve)

✔ Strong focus on yearly seasonality

✔ Lag & rolling statistical feature engineering

✔ Supervised learning formulation for forecasting

✔ Linear Regression model outperforming baselines

✔ Residual analysis for model validation

✔ Command Line Interface (CLI) for forecasting

✔ Stateful, recursive multi-step predictions

📂 Dataset

Source: Walmart Sales Forecasting Dataset (Kaggle)

Key Files Used:

train.csv – historical weekly sales (target: Weekly_Sales)

features.csv – external factors (markdowns, CPI, unemployment, holidays)

stores.csv – store metadata

Time Range:
📅 Feb 2010 – Oct 2012
⏱ Weekly frequency

🧠 Problem Framing

Given historical weekly sales data, can we accurately forecast future sales while capturing strong seasonal and demand patterns?

This problem is highly relevant for:

Retail demand planning

Inventory management

Revenue forecasting

Supply chain optimization

📊 Exploratory Time Series Analysis

Key findings:

Strong yearly seasonality

Sales spikes around holiday periods

Stable seasonal patterns across years

No missing weekly observations

These insights motivated the use of seasonal baselines and lag-based features.

📈 Baseline Models

The following baselines were evaluated using Mean Absolute Error (MAE):

Model	MAE
Naïve (last week = next week)	~3.14M
Seasonal Naïve (same week last year)	~1.43M

📌 The seasonal naïve model reduced error by ~55%, confirming strong yearly seasonality.

⚙️ Feature Engineering

To convert the forecasting task into a supervised learning problem, the following features were engineered:

🔁 Lag Features

lag_1 – last week

lag_2 – two weeks ago

lag_4 – four weeks ago

lag_52 – same week last year

📊 Rolling Statistics

rolling_mean_4 – 4-week rolling mean

rolling_mean_12 – 12-week rolling mean

These features allow the model to combine short-term momentum with long-term seasonality.

🤖 Model: Linear Regression

A Linear Regression model was trained using a time-based train/test split (last 6 months held out).

📉 Performance
Model	MAE
Seasonal Naïve	~1.43M
Linear Regression	~0.89M ✅

📌 ~37% improvement over the seasonal baseline

🔍 Model Interpretation

lag_52 had the largest absolute coefficient

Confirms that yearly seasonality is the dominant driver

Short-term lags improve responsiveness to local demand changes

This aligns strongly with retail business intuition.

🧪 Residual Analysis

Residual diagnostics showed:

Errors centered around zero (unbiased model)

Larger residuals during extreme seasonal peaks

Strong performance during normal demand periods

📌 Indicates a well-behaved and reliable forecasting model.

----------------------------------------------------

🖥️ Command Line Interface (CLI)

A CLI was built to simulate real-world forecasting workflows.

🔧 Supported Commands

Train the model:

python forecast.py train


Forecast next week:

python forecast.py predict


Forecast multiple weeks ahead:

python main.py predict --weeks 4

----------------------------------------------------

🔄 Stateful Forecasting

Uses the last 52 weeks of sales

Updates historical state after each prediction

Enables recursive multi-step forecasting

🏗 Project Structure
sales-forecasting-time-series/
│
├── main.py          # Training & CLI forecasting logic
├── model.pkl            # Trained model (ignored in Git)
├── last_52_weeks.csv    # Rolling historical state
├── data/
│   ├── train.csv
│   ├── features.csv
│   └── stores.csv
├── README.md
└── requirements.txt

📌 Key Learnings

Seasonal baselines are critical in time series

Complex models are useless without strong baselines

Lag & rolling features unlock powerful forecasting performance

Time-aware evaluation prevents data leakage

Residual analysis is essential for trust and interpretability

Forecasting systems require state, not just models

🧠 Skills Demonstrated

Time Series Analysis

Forecasting & Baseline Modeling

Feature Engineering

Supervised Learning

Model Evaluation (MAE)

Residual Diagnostics

Python, Pandas, Scikit-learn

CLI Development

Production-aware ML thinking

📬 Author

Humza
Data Science & Machine Learning Enthusiast

If you found this project useful or insightful, feel free to ⭐ the repository.