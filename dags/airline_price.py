from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

import pandas as pd
from scipy.stats import yeojohnson
import sklearn.preprocessing as preproc
import os

def data_pre(ti, path="/opt/airflow/data/raw_data/Clean_Dataset.csv"):
    data = pd.read_csv(path, index_col=0)
    # Encode the ordinal variables "stops" and "class".
    data["stops"] = (
        data["stops"].replace({"zero": 0, "one": 1, "two_or_more": 2}).astype(int)
    )
    data["class"] = data["class"].replace({"Economy": 0, "Business": 1}).astype(int)
    dummies_variables = [
        "airline",
        "source_city",
        "destination_city",
        "departure_time",
        "arrival_time",
    ]
    # one-hot encoding
    dummies = pd.get_dummies(data[dummies_variables])
    data = pd.concat([data, dummies], axis=1)
    data = data.drop(
        [
            "flight",
            "airline",
            "source_city",
            "destination_city",
            "departure_time",
            "arrival_time",
        ],
        axis=1,
    )
    print(data.head())
    # apply transformer for normal distribution
    col = "duration"
    y_value, _ = yeojohnson(data[col])
    data[col] = y_value
    # Standardization
    cols = ["duration", "days_left"]
    data[cols] = preproc.StandardScaler().fit_transform(data[cols])
    print(data.head())

    # outlier economy
    price = data[data["class"] == 0].price
    lower_limit = price.mean() - 3 * price.std()
    upper_limit = price.mean() + 3 * price.std()
    print("economy: ")
    print(lower_limit)
    print(upper_limit)
    # economy class data index
    cls_eco = data[
        (data["class"] == 0)
        & (data["price"] >= lower_limit)
        & (data["price"] <= upper_limit)
    ].index
    # outlier business
    price = data[data["class"] == 1].price
    lower_limit = price.mean() - 3 * price.std()
    upper_limit = price.mean() + 3 * price.std()
    print("Business:")
    print(lower_limit)
    print(upper_limit)
    # business class data index
    cls_bsn = data[
        (data["class"] == 1)
        & (data["price"] >= lower_limit)
        & (data["price"] <= upper_limit)
    ].index
    try:
        os.makedirs("/opt/airflow/data/feature/")
    except:
        pass

    data.iloc[cls_eco].to_csv("/opt/airflow/data/feature/eco_features.csv", index=False)
    data.iloc[cls_bsn].to_csv("/opt/airflow/data/feature/bus_features.csv", index=False)

    context_dict = {
        "business": "/opt/airflow/data/feature/bus_features.csv",
        "economy": "/opt/airflow/data/feature/eco_features.csv",
    }
    ti.xcom_push(key="data_preparation_context", value=context_dict)
    return None


airline_dag = DAG(
    "Airline_ticket_price_prediction_DAG",
    schedule_interval="@daily",
    start_date=datetime(2023, 8, 18),
)

with airline_dag:
    data_preparation_task = PythonOperator(
        task_id="data_preparation", python_callable=data_pre, provide_context=True,
    )
