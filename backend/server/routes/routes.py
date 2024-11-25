import logging
import os
import sys
import json

sys.path.append('/app/server')

# from ResnetNetwork import *  # for local testing
# from ecg_data_pb2 import AbdominalData, ChestData, CapturedECGData  # for local testing
from ..ResnetNetwork import *  # for docker
from ..ecg_data_pb2 import AbdominalData, ChestData, CapturedECGData  # for docker
from flask import Blueprint, jsonify, request, Response, stream_with_context

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
        # model_path = f"db/models/last_model_2024-11-16.pt"  # for local testing
        model_path = os.path.join(os.getcwd(), f"server/db/models/last_model_2024-11-16.pt")  # model.module for docker
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
    EXPECTED_SIZE = 8226  # Size threshold in bytes
    global model
    device = torch.device('cpu')

    # try:
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    def stream_responses():
        buffer = b''  # Initialize an empty buffer to accumulate chunks

        for chunk in request.stream:
            buffer += chunk  # Add chunk to the buffer

            # Process complete messages from the buffer
            while len(buffer) >= EXPECTED_SIZE:
                try:
                    # Extract a single complete message
                    message_bytes = buffer[:EXPECTED_SIZE]
                    buffer = buffer[EXPECTED_SIZE:]  # Retain extra data for the next message

                    # Parse the Protobuf message
                    ecg_data = CapturedECGData()
                    ecg_data.ParseFromString(message_bytes)

                    # Process the parsed message
                    response, status_code = process_ecg_data(
                        ecg_data.abdominal_data.values,
                        ecg_data.chest_data.values,
                        ecg_data.timestamp
                    )
                    if status_code == 200:
                        yield f"data: {response.get_data(as_text=True)}\n\n"
                    else:
                        logging.debug(f"Error response: {response.get_data(as_text=True)}")  # Debug log
                        yield f"data: {response.get_data(as_text=True)}\n\n"

                except Exception as e:
                    logging.debug("Protobuf Parsing Error:", str(e))
                    yield f"data: {jsonify({'error': 'Failed to parse Protobuf message'}).get_data(as_text=True)}\n\n"

        # Handle any remaining incomplete message in the buffer
        if len(buffer) > 0:
            logging.debug(f"Incomplete message remaining in buffer: {len(buffer)} bytes")
            yield f"data: {jsonify({'error': 'Incomplete Protobuf message received'}).get_data(as_text=True)}\n\n"

    # Return a stream response using server-sent events
    return Response(stream_with_context(stream_responses()), content_type='text/event-stream')


def process_ecg_data(abdominal_data, chest_data, timestamp):
    global model
    device = torch.device('cpu')

    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    if len(abdominal_data) != 1024 or len(chest_data) != 1024:
        return jsonify({"error": "Each data array must contain exactly 1024 elements"}), 500

    # Prepare tensors outside the try block
    abdominal_tensor = torch.tensor(abdominal_data).view(1, 1, 1024).float().to(device)
    chest_tensor = torch.tensor(chest_data).view(1, 1, 1024).float().to(device)
    input_tensor = torch.cat((abdominal_tensor, chest_tensor), dim=1)  # Shape: [1, 2, 1024]

    try:
        # Perform inference inside the try block
        with torch.no_grad():
            model_output = model(input_tensor)

        fetal_ecg, tensor2, maternal_ecg, tensor4 = model_output

        # Return results
        return jsonify({
            "timestamp_data": timestamp,
            "fetal_ecg_data": fetal_ecg.cpu().numpy().tolist(),
            "tensor2_data": tensor2.cpu().numpy().tolist(),
            "maternal_ecg_data": maternal_ecg.cpu().numpy().tolist(),
            "tensor4_data": tensor4.cpu().numpy().tolist()
        }), 200

    except Exception as e:
        # Handle only inference-related errors here
        return jsonify({"error": str(e)}), 500
