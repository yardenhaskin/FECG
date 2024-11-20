import logging
import os
import sys

sys.path.append('/app/server')

# from ResnetNetwork import * # for local testing
# from ecg_data_pb2 import AbdominalData, ChestData, CapturedECGData # for local testing
from ..ResnetNetwork import * # for docker
from ..ecg_data_pb2 import AbdominalData, ChestData, CapturedECGData # for docker
from flask import Blueprint, jsonify



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
        # model_path = f"db/models/last_model_2024-11-16.pt" # for local testing
        model_path = os.path.join(os.getcwd(), f"server/db/models/last_model_2024-11-16.pt") # model.module
        model = torch.load(model_path, map_location=torch.device('cpu'))

        return jsonify({
            "status": "success",
            "message": "Model loaded successfully.",
            "model_id": f"last_model_2024-11-16"
        }), 200
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return jsonify({"error": str(e)}), 500


@bp.route('/separate-ecg', methods=['POST'])
def separate_ecg():
    global model
    device = torch.device('cpu')

    try:
        if model is None:
            return jsonify({"error": "Model is not loaded"}), 500
        
        # If the model was wrapped with DistributedDataParallel, unwrap it by accessing `model.module`
        # if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        #     model = model.module

        # Alternatively, load the model without any parallel wrapper if distributed is not needed
        # model = model.to(device)

        abdominal_arr = [0.0] * 1024
        chest_arr = [0.0] * 1024

        if len(abdominal_arr) != 1024 or len(chest_arr) != 1024:
            return jsonify({"error": "Each data array must contain exactly 1024 elements"}), 400

        # Create a tensor with 2 channels: abdominal and chest ECG data
        abdominal_tensor = torch.tensor(abdominal_arr).view(1, 1, 1024).float().to(device)
        chest_tensor = torch.tensor(chest_arr).view(1, 1, 1024).float().to(device)

        # Concatenate along the channel dimension to create a 2-channel input
        input_tensor = torch.cat((abdominal_tensor, chest_tensor), dim=1)  # Shape: [1, 2, 1024]

        # model.eval()

        with torch.no_grad():
            # Perform inference on the model
            model_output = model(input_tensor)

        fetal_ecg, tensor2, maternal_ecg, tensor4 = model_output

        return jsonify({
            "fetal_ecg_data": fetal_ecg.cpu().numpy().tolist(),
            "tensor2_data": tensor2.cpu().numpy().tolist(),
            "maternal_ecg_data": maternal_ecg.cpu().numpy().tolist(),
            "tensor4_data": tensor4.cpu().numpy().tolist()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
