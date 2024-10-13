import joblib
import torch
from flask import Blueprint, request, jsonify
from ..ecg_data_pb2 import AbdominalData, ChestData, CapturedECGData

abdominal_data = AbdominalData()

bp = Blueprint('routes', __name__)

# Global variable to store the loaded model
model = None


def is_valid_protobuf(data, field, protobuf_class):
    if field in data:
        if not isinstance(data[field], protobuf_class):
            return False, f"{field} must be a {protobuf_class.__name__} instance"
    return True, ""


def validate_data_train(data):
    if not data or 'abdominal_data' not in data or 'chest_data' not in data or 'timestamp' not in data:
        return False, "Invalid request parameters"

    for field, protobuf_class in [('abdominal_data', AbdominalData), ('chest_data', ChestData)]:
        is_valid, error_message = is_valid_protobuf(data, field, protobuf_class)
        if not is_valid:
            return False, error_message

    return True, ""


@bp.route('/load-model', methods=['POST'])
def load_model():
    global model

    print(f"CUDA available: {torch.cuda.is_available()}")

    try:
        # Load the pre-trained model based on the id
        model_path = f"server/db/models/resnet_model_0.pkl"
        model = torch.load(model_path, map_location=torch.device('cpu'))

        return jsonify({
            "status": "success",
            "message": "Model loaded successfully.",
            "model_id": f"resnet_model_0.pkl"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/separate-ecg', methods=['POST'])
def separate_ecg():
    global model
    data = request.get_json()
    is_valid, error_message = validate_data_train(data)
    if not is_valid:
        return jsonify({"error": error_message}), 400

    try:
        # Placeholder for ECG signal separation logic
        if model is None or not hasattr(model, 'fit'):
            return jsonify({"error": "Model is not loaded or does not support the fit method"}), 500
        model.fit(data['abdominal_data'], data['chest_data'])
        # fetal_ecg, maternal_ecg = separate_signals(data['abdominal_data'], data['chest_data'])
        return jsonify({
            "timestamp_data": "TBD",
            "fetal_ecg_data": "TBD",
            "maternal_ecg_data": "TBD"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
