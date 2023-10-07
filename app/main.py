from flask import Flask, request, jsonify
import mlflow
import numpy as np

# model details
logged_model = "s3://t-test-bucket-v1/artifacts/1/68335b098ea84607a67f2efb67198643/artifacts/KNNEconomy_exp"
loaded_model = mlflow.pyfunc.load_model(logged_model)

app = Flask(__name__)


@app.route("/api/v1", methods=["POST"])
def pred():
    # Get the JSON data from the request
    request_data = request.json
    # Perform any processing you want with the received data
    if "features" in request_data:
        features = request_data["features"]
        prediction = loaded_model.predict(np.array(features).reshape(1, -1))
        resp = {
            "prediction": prediction[0],
            "run_id": loaded_model.metadata.run_id,
            "model_uuid": loaded_model.metadata.model_uuid,
            "utc_time_created": loaded_model.metadata.utc_time_created,
        }
        return jsonify(resp), 200
    else:
        return jsonify({"error": "features missing"}), 400


@app.route("/health", methods=["GET"])
def heath():
    return jsonify({"message": "api active!"}), 200
