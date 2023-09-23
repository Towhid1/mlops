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

# mlflow import
import mlflow
from mlflow import log_metric, log_param

# evidently ai import for reporting/monitoring
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report

# MLflow stuff
TRACKING_SERVER_HOST = 'mlflow'


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
    data.iloc[cls_bsn].to_csv("/opt/airflow/data/feature/bsn_features.csv", index=False)

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
    global TRACKING_SERVER_HOST
    exmperiment_name = "Business_exp"
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_experiment(exmperiment_name)
    model_name = "KNN"
    dir_dict = ti.xcom_pull(key='data_preparation_context')
    filename = dir_dict['business']
    feature = pd.read_csv(filename)
    target = feature.pop("price")
    x_train, x_test, y_train, y_test = train_test_split(
        feature, target, random_state=1, test_size=0.3,
        shuffle=True)

    with mlflow.start_run():
        model = KNeighborsRegressor()
        trained_model = model.fit(x_train, y_train)
        y_pred = trained_model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        log_param("model_name", model_name)
        log_param("one-hot-encoding", True)
        log_param("outlier", True)
        log_param("transformer", "yeojhonson")
        log_metric("MAE", mae)
        model_info = mlflow.sklearn.log_model(trained_model, model_name+exmperiment_name)
        reg_model = mlflow.register_model(model_info.model_uri, model_name+exmperiment_name)
    mlflow.end_run()
    print(f"model reg : {reg_model}")
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
    global TRACKING_SERVER_HOST
    exmperiment_name = "Economy_exp"
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_experiment(exmperiment_name)
    model_name = "KNN"

    dir_dict = ti.xcom_pull(key='data_preparation_context')
    filename = dir_dict['economy']
    feature = pd.read_csv(filename)
    target = feature.pop("price")
    x_train, x_test, y_train, y_test = train_test_split(
        feature, target, random_state=1, test_size=0.3,
        shuffle=True)

    with mlflow.start_run():
        model = KNeighborsRegressor()
        trained_model = model.fit(x_train, y_train)
        y_pred = trained_model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        log_param("model_name", model_name)
        log_param("one-hot-encoding", True)
        log_param("outlier", True)
        log_param("transformer", "yeojhonson")
        log_metric("MAE", mae)
        model_info = mlflow.sklearn.log_model(trained_model, model_name+exmperiment_name)
        reg_model = mlflow.register_model(model_info.model_uri, model_name+exmperiment_name)

    mlflow.end_run()
    print(f"model reg : {reg_model}")
    print(f"economy class mean_absolute_error: {mae}")
    return None


def data_drift(ti):
    dir_dict = ti.xcom_pull(key='feature_dir')
    dir_list = ['economy', 'business']
    for key in dir_list:
        filename = dir_dict[key]
        print(filename)
        data_frame = pd.read_csv(filename)
        data_frame = data_frame.drop(columns=['class'])
        data_frame_ref = data_frame.sample(n=500, replace=False)
        data_frame_cur = data_frame.sample(n=500, replace=False)
        print(data_frame.columns)

        # Get categorical columns (object or categorical dtype)
        categorical_cols = data_frame.select_dtypes(
            include=['bool', 'object', 'category']
        ).columns.tolist()

        data_columns = ColumnMapping()
        data_columns.categorical_features = categorical_cols
        data_columns.numerical_features = ['stops', 'duration', 'days_left']
        data_columns.target = 'price'

        # data_columns.prediction = 'prediction'
        dir_path = "/opt/airflow/data/reports/"
        try:
            os.makedirs(dir_path)
        except:
            pass
        file_path = f"data_drift_{key}.html"
        # roprting code
        data_drift_report = Report(metrics=[DataDriftPreset()])
        data_drift_report.run(
            reference_data=data_frame_ref,
            current_data=data_frame_cur,
            column_mapping=data_columns
        )
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        data_drift_report.save_html(os.path.join(dir_path, file_path))


airline_dag = DAG(
    "Airline_ticket_price_prediction_DAG",
    # schedule_interval="@daily",
    schedule_interval=None,
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

    data_drift_task = PythonOperator(
        task_id="data_drift_task_id", python_callable=data_drift, provide_context=True
    )

    data_preparation_task >> data_drift_task >> [bsn_training_task, eco_training_task]  # pylint: disable=w0104
