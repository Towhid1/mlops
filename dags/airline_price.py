# airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

# import for training
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

# data preparation
import pandas as pd
from scipy.stats import yeojohnson
import sklearn.preprocessing as preproc
import os


def data_pre(ti, path="/opt/airflow/data/raw_data/Clean_Dataset.csv"):
    """
    Prepare and preprocess the dataset for training.

    This function reads a CSV file located at the given path, performs various data
    preprocessing steps, including label-encoding, one-hot encoding, transformation for normal distribution,
    standardization, and outlier removal. It saves the preprocessed data for economy and business
    class separately and provides context information for downstream tasks.

    Args:
        ti (TaskInstance): The Airflow TaskInstance object.
        path (str, optional): Path to the CSV file containing the raw dataset. Default is
            "/opt/airflow/data/raw_data/Clean_Dataset.csv".

    Returns:
        None
    """
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
        "business": "/opt/airflow/data/feature/bsn_features.csv",
        "economy": "/opt/airflow/data/feature/eco_features.csv",
    }
    ti.xcom_push(key="data_preparation_context", value=context_dict)
    return None


def bsn_training(ti):
    """
    Perform training and evaluation of a K-Nearest Neighbors Regressor model
    for predicting prices using business-class features.

    Args:
        ti (TaskInstance): The Airflow TaskInstance object.

    Returns:
        None
    """
    dir_dic = ti.xcom_pull(key='data_preparation_context')
    filename = dir_dic['business']
    feature = pd.read_csv(filename)
    target = feature.pop("price")
    x_train, x_test, y_train, y_test = train_test_split(
        feature, target, random_state=1, test_size=0.3,
        shuffle=True)

    model = KNeighborsRegressor()
    trained_model = model.fit(x_train, y_train)
    y_pred = trained_model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"business class mean_absolute_error: {mae}")
    return None


def eco_training(ti):
    """
    Perform training and evaluation of a K-Nearest Neighbors Regressor model
    for predicting prices using economy-class features.

    Args:
        ti (TaskInstance): The Airflow TaskInstance object.

    Returns:
        None
    """
    dir_dic = ti.xcom_pull(key='data_preparation_context')
    filename = dir_dic['economy']
    feature = pd.read_csv(filename)
    target = feature.pop("price")
    x_train, x_test, y_train, y_test = train_test_split(
        feature, target, random_state=1, test_size=0.3,
        shuffle=True)

    model = KNeighborsRegressor()
    trained_model = model.fit(x_train, y_train)
    y_pred = trained_model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"economy class mean_absolute_error: {mae}")
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

    eco_training_task = PythonOperator(
        task_id="eco_training_task", python_callable=eco_training, provide_context=True
    )

    bsn_training_task = PythonOperator(
        task_id="bsn_training_task", python_callable=bsn_training, provide_context=True
    )

    data_preparation_task >> [bsn_training, eco_training]
