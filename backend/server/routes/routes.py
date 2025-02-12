import os
import sys
import time
import logging
import threading
import torch
import orjson
from flask import Blueprint, jsonify, request, Response, stream_with_context

# Import local modules
from ResnetNetwork import *  # for local testing
from ecg_data_pb2 import AbdominalData, ChestData, CapturedECGData  # for local testing

# Import docker modules
# from ..ResnetNetwork import *  # for Docker
# from ..ecg_data_pb2 import AbdominalData, ChestData, CapturedECGData  # for Docker

# Append the server directory to sys.path
sys.path.append('/app/server')

# Initialize Flask Blueprint
bp = Blueprint('routes', __name__)

global_model = None
PROTOBUF_MESSAGE_SIZE_BYTES = 8226
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = torch.empty((1, 2, 1024), dtype=torch.float32, device=DEVICE)

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def keep_gpu_warm():
    """Keeps the GPU active by running periodic inference."""
    global global_model
    while True:
        try:
            if global_model is not None:
                with torch.no_grad():
                    _ = global_model(torch.randn(1, 2, 1024, dtype=torch.float32, device=DEVICE))
            time.sleep(0.05)  # Adjust based on load
        except Exception as e:
            print(f"GPU warm-up thread error: {e}")  # Log errors instead of silent failure


# Start the warm-up thread only after model initialization
def start_gpu_warmup():
    warmup_thread = threading.Thread(target=keep_gpu_warm, daemon=True)
    warmup_thread.start()


def is_valid_protobuf(data, field, protobuf_class):
    if field in data:
        if not isinstance(data[field], protobuf_class):
            return False, f"{field} must be a {protobuf_class.__name__} instance"
    return True, ""


def validate_data_train(data):
    # Check if all required fields are present
    missing_fields = [field for field in ['abdominal_data', 'chest_data', 'timestamp'] if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"

    # Validate individual fields using Protobuf classes
    for field, protobuf_class in [('abdominal_data', AbdominalData), ('chest_data', ChestData)]:
        is_valid, error_message = is_valid_protobuf(data, field, protobuf_class)
        if not is_valid:
            return False, error_message

    # Validate timestamp format (basic example, can be enhanced)
    try:
        time.strptime(data['timestamp'], "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False, "Invalid timestamp format, expected ISO 8601 (e.g., '2024-10-27T12:00:00Z')"

    return True, ""


@bp.route('/user/<id>', methods=['GET'])
def get_user(id):
    try:
        # Path to the users directory
        user_file_path = os.path.join(os.getcwd(), 'server/db/users/users.json')

        # Read the content of the user file
        with open(user_file_path, 'r') as file:
            users_data = orjson.loads(file)

        # Find the user with the matching id
        user_data = next((user for user in users_data['users'] if user['id'] == id), None)

        if user_data is None:
            return orjson.dumps({"error": "User not found"}), 404

        return jsonify({"user_data": user_data}), 200
    except Exception as e:
        return orjson.dumps({"error": str(e)}), 500


@bp.route('/user/<id>', methods=['POST'])
def modify_user(id):
    try:
        data = request.get_json()
        # TODO: Validate the request JSON payload
        # TODO: parse the request JSON payload
        # Path to the users directory
        user_file_path = os.path.join(os.getcwd(), f"server/db/users/users.json")

        with open(user_file_path, 'r') as file:
            users_data = orjson.loads(file)

        # Find the user with the matching id
        user_data = next((user for user in users_data['users'] if user['id'] == id), None)

        if user_data is None:
            # TODO: Add the user to the users file
            return orjson.dumps({"Success: User added successfully"}), 200
        else:
            # TODO: Update the user file
            return orjson.dumps({"Success: User updated successfully"}), 200

    except Exception as e:
        return orjson.dumps({"error": str(e)}), 500


@bp.route('/user/<id>', methods=['DELETE'])
def delete_user(id):
    try:
        # Path to the users directory
        user_file_path = os.path.join(os.getcwd(), 'server/db/users/users.json')

        # TODO: also delete model

        with open(user_file_path, 'r') as file:
            users_data = orjson.loads(file)

        # Find the user with the matching id
        user_data = next((user for user in users_data['users'] if user['id'] == id), None)

        if user_data is None:
            return orjson.dumps({"error": "User not found"}), 404
        else:
            # Remove the user from the list
            users_data['users'] = [user for user in users_data['users'] if user['id'] != id]

            # Write the updated data back to the file
            with open(user_file_path, 'w') as file:
                orjson.dumps(users_data, file)

            return orjson.dumps({"success": "User deleted successfully"}), 200

    except Exception as e:
        return orjson.dumps({"error": str(e)}), 500


@bp.route('/models', methods=['GET'])
def get_all_models_names():
    try:
        # Path to the models directory
        models_dir = os.path.join(os.getcwd(), 'server/db/models')

        # List all files in the models directory
        model_files = os.listdir(models_dir)

        # Filter out non-model files if necessary (e.g., by extension)
        model_names = [f for f in model_files if f.endswith('.pt')]

        return orjson.dumps({"models": model_names}), 200
    except Exception as e:
        return orjson.dumps({"error": str(e)}), 500


@bp.route('/size', methods=['GET'])  # Get the size of the protobuf message
def get_size():
    ecg_sample = CapturedECGData(
        abdominal_data=AbdominalData(values=[0.0] * 1024),
        chest_data=ChestData(values=[0.0] * 1024),
        timestamp="2024-10-27T12:00:00Z"
    )
    message_size = len(ecg_sample.SerializeToString())
    return orjson.dumps({"message_size": message_size}), 200


@bp.route('/load-model', methods=['POST'])
def load_model():
    global global_model

    try:
        # Get the model id from the request JSON payload
        data = request.get_json()
        model_id = data.get('id')  # Default model id if not provided

        if model_id is None:
            return orjson.dumps({"error": "ID doesnt match a model"}), 400

        # Load the pre-trained model based on the id
        if False:
            # TODO: check if id exists
            model_path = f"db/models/{model_id}.pt"  # for local testing
            # model_path = os.path.join(os.getcwd(), f"server/db/models/{model_id}.pt")
        else:
            model_path = f"db/models/base_model_16-11-24.pt"  # for local testing
            # model_path = os.path.join(os.getcwd(), f"db/models/base_model_16-11-24.pt".pt")
        if torch.cuda.is_available():
            logging.info("CUDA is enabled.")
        else:
            logging.info("CUDA is not available")

        global_model = torch.load(model_path, map_location=DEVICE)

        # Warm-up GPU
        # warmup_model(model, DEVICE)
        start_gpu_warmup()  # Start GPU warm-up thread

        return orjson.dumps({
            "status": "success",
            "message": "Model loaded successfully.",
            "model_id": model_id
        }), 200
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return orjson.dumps({"error": str(e)}), 500


@bp.route('/separate-ecg', methods=['POST'])
def separate_ecg():
    start_time = time.time()  # Start timer
    global global_model

    if global_model is None:
        return orjson.dumps({"error": "Model is not loaded"}), 500

    try:
        # Read the entire request body
        message_bytes = request.get_data()

        # Process the Protobuf message
        response = process_chunk(message_bytes)
        if response[1] != 200:
            logging.error(f"Error processing Protobuf message: {response[0]}")
            return response

        # Calculate and log total processing time
        total_time = time.time() - start_time
        logging.info(f"Total processing time for /separate-ecg-v2: {total_time:.6f} seconds")
        return response

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return orjson.dumps({"error": str(e)}), 500


def process_chunk(message_bytes):
    try:
        # Parse the Protobuf message
        ecg_data = CapturedECGData()
        ecg_data.ParseFromString(message_bytes)

        # Validate the parsed data
        data = {
            'abdominal_data': ecg_data.abdominal_data,
            'chest_data': ecg_data.chest_data,
            'timestamp': ecg_data.timestamp,
        }
        is_valid, error_message = validate_data_train(data)
        if not is_valid:
            return orjson.dumps({'error': error_message}), 400

        # Process the parsed message
        return process_ecg_data(
            ecg_data.abdominal_data.values,
            ecg_data.chest_data.values,
            ecg_data.timestamp
        )

    except Exception as e:
        logging.error(f"Failed to parse Protobuf message: {str(e)}")
        return orjson.dumps({'error': 'Failed to parse Protobuf message'}), 400


def process_ecg_data(abdominal_data, chest_data, timestamp):
    global global_model, input_tensor

    if global_model is None:
        return orjson.dumps({"error": "Model is not loaded"}), 500

    if len(abdominal_data) != 1024 or len(chest_data) != 1024:
        return orjson.dumps({"error": "Each data array must contain exactly 1024 elements"}), 500

    # Copy new data into the pre-allocated tensor (avoids reallocation)
    input_tensor[0, 0].copy_(torch.as_tensor(abdominal_data, dtype=torch.float32, device=DEVICE))
    input_tensor[0, 1].copy_(torch.as_tensor(chest_data, dtype=torch.float32, device=DEVICE))

    # TODO: Run inference with the loaded model on the actual data and use the trained model to process the data (save? output?)
    try:
        # Start the timer
        start_time = time.time()
        # Perform inference inside the try block
        with torch.no_grad():
            fetal_ecg, tensor2, maternal_ecg, tensor4 = global_model(input_tensor)

        # End the timer
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        logging.info(f"Processing time for a single input: {elapsed_time:.6f} seconds")

        # Return results
        return orjson.dumps({
            "timestamp_data": timestamp,
            "fetal_ecg_data": fetal_ecg.cpu().numpy().tolist(),
            "tensor2_data": tensor2.cpu().numpy().tolist(),
            "maternal_ecg_data": maternal_ecg.cpu().numpy().tolist(),
            "tensor4_data": tensor4.cpu().numpy().tolist()
        }), 200

    except Exception as e:
        # Handle only inference-related errors here
        return orjson.dumps({"error": str(e)}), 500
