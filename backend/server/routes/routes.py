import logging
import os
import sys

sys.path.append('/app/server')

from ..ResnetNetwork import *
from flask import Blueprint, jsonify
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


@bp.route('/size', methods=['GET'])  # Get the size of the protobuf message
def get_size():
    ecg_sample = CapturedECGData(
        abdominal_data=AbdominalData(values=[0.0] * 1024),
        chest_data=ChestData(values=[0.0] * 1024),
        timestamp="2024-10-27T12:00:00Z"
    )
    message_size = len(ecg_sample.SerializeToString())
    return jsonify({"message_size": message_size}), 200


logging.basicConfig(level=logging.DEBUG)


@bp.route('/load-model', methods=['POST'])
def load_model():
    global model

    try:
        # Load the pre-trained model based on the id
        model_path = os.path.join(os.getcwd(), f"server/db/models/last_model_2024-11-05.pth")
        model = torch.load(model_path, map_location=torch.device('cpu'))

        return jsonify({
            "status": "success",
            "message": "Model loaded successfully.",
            "model_id": f"best_model_real_200_epochs_cont.pkl"
        }), 200
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return jsonify({"error": str(e)}), 500


@bp.route('/separate-ecg', methods=['POST'])
def separate_ecg():
    global model
    # data = request.get_json()
    # if not data:
    #     return jsonify({"error": "No data provided"}), 400

    try:
        if model is None:
            return jsonify({"error": "Model is not loaded"}), 500

        # Extract abdominal and chest data
        # abdominal_arr = data.get('abdominal_data', [])
        # chest_arr = data.get('chest_data', [])
        # timestamp = data.get('timestamp', "")

        abdominal_arr = [0.0] * 1024
        chest_arr = [0.0] * 1024

        # Validate data length
        if len(abdominal_data) != 1024 or len(chest_arr) != 1024:
            return jsonify({"error": "Each data array must contain exactly 1024 elements"}), 400

        # Create tensor with 8 copies of abdominal_data and 8 copies of chest_data
        abdominal_tensor = torch.tensor([abdominal_arr] * 8).view(8, 1024, 1).float()
        chest_tensor = torch.tensor([chest_arr] * 8).view(8, 1024, 1).float()

        # Concatenate to form a [16, 1024, 1] tensor
        input_tensor = torch.cat((abdominal_tensor, chest_tensor), dim=0)

        # Ensure the model can handle the input
        with torch.no_grad():
            model_output = model(input_tensor)

        # Assuming model output includes fetal and maternal ECG tensors
        fetal_ecg, maternal_ecg = model_output

        return jsonify({
            # "timestamp": timestamp,
            "fetal_ecg_data": fetal_ecg.numpy().tolist(),
            "maternal_ecg_data": maternal_ecg.numpy().tolist()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
