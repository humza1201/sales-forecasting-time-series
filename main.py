import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import argparse

MODEL_FILE = 'model.pkl'

parser = argparse.ArgumentParser(description="Sales Forecasting CLI")
parser.add_argument(
    "command",
    choices=["train","predict"],
    help="Command to run"
)

parser.add_argument(
    "--weeks",
    type=int,
    default=1,
    help="Number of weeks to forecast"
)

args = parser.parse_args()

if args.command == "train":
    data = pd.read_csv('features.csv')
    df_features = pd.DataFrame(data)

    stores = pd.read_csv('stores.csv')
    df_stores = pd.DataFrame(stores)

    train_data = pd.read_csv('train.csv')
    df_train = pd.DataFrame(train_data)

    df_features['Date'] = pd.to_datetime(df_features['Date'])
    df_train['Date'] = pd.to_datetime(df_train['Date'])

    df = df_train.merge(df_features , on= ['Store', 'Date', 'IsHoliday'], how='left')
    df = df.merge(df_stores, on=['Store'], how='left')
    

    weekly_total = df.groupby('Date')['Weekly_Sales'].sum().sort_index()
    weekly_total.tail(52).reset_index().to_csv("last_52_weeks.csv", index=False)

    timeseries_df = pd.DataFrame({'y': weekly_total})
    for lag in [1, 2, 4, 52]:
        timeseries_df[f'lag_{lag}'] = timeseries_df['y'].shift(lag)
    timeseries_df['rolling_mean_4'] = timeseries_df['y'].rolling(window=4).mean().shift(1)
    timeseries_df['rolling_mean_12'] = timeseries_df['y'].rolling(window=12).mean().shift(1)
    timeseries_df = timeseries_df.dropna()

    split_dates = timeseries_df.index.max() - pd.DateOffset(months = 6)
    train_df = timeseries_df[timeseries_df.index <= split_dates]
    test_df = timeseries_df[timeseries_df.index > split_dates]

    x_train = train_df.drop(columns='y')
    y_train = train_df['y']

    x_test = test_df.drop(columns='y')
    y_test = test_df['y']
    
    timeseries_df.to_csv("timeserires.csv")

    reg = LinearRegression()
    reg.fit(x_train, y_train)
    joblib.dump(reg, MODEL_FILE)
    print('model trained successfully!')

elif args.command == "predict":
    reg = joblib.load(MODEL_FILE)
    
    history = pd.read_csv("last_52_weeks.csv", parse_dates=["Date"])
    
    for step in range(args.weeks):
        history = history.sort_values("Date")
        
        last_values = history["Weekly_Sales"]
        
        features = {
            "lag_1": last_values.iloc[-1],
            "lag_2": last_values.iloc[-2],
            "lag_4": last_values.iloc[-4],
            "lag_52": last_values.iloc[-52],
            "rolling_mean_4": last_values.iloc[-4:].mean(),
            "rolling_mean_12": last_values.iloc[-12:].mean()
        }
        
        X_next = pd.DataFrame([features])
        next_week_prediction = reg.predict(X_next)[0]
        
        new_row = {
            "Date": history["Date"].max() + pd.Timedelta(weeks=1),
            "Weekly_Sales": next_week_prediction
        }

        history = pd.concat([history, pd.DataFrame([new_row])])
        history = history.tail(52)

        history.to_csv("last_52_weeks.csv", index=False)


        print(f"Forecasted sales for {step + 1} week: {next_week_prediction:,.2f}")