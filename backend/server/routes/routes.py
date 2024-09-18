from flask import Blueprint, request, jsonify

bp = Blueprint('routes', __name__)


def is_non_empty_list(data, field):
    if field in data:
        if not isinstance(data[field], list):
            return False, f"{field} must be a list"
        if not data[field]:
            return False, f"{field} must be a non-empty list"
    return True, ""


def validate_data_train(data):
    if not data or 'abdominal_data' not in data or 'chest_data' not in data or 'timestamp' not in data:
        return False, "Invalid request parameters"

    for field in ['abdominal_data', 'chest_data']:
        is_valid, error_message = is_non_empty_list(data, field)
        if not is_valid:
            return False, error_message

    return True, ""


def validate_data_load(data):
    if not data:
        return False, "Invalid request parameters: data is missing"

    for field in ['abdominal_data', 'chest_data']:
        is_valid, error_message = is_non_empty_list(data, field)
        if not is_valid:
            return False, error_message

    return True, ""


@bp.route('/train-model', methods=['POST'])
def train_model():
    data = request.get_json()
    is_valid, error_message = validate_data_train(data)
    if not is_valid:
        return jsonify({"error": error_message}), 400

    try:
        # Placeholder for model training logic
        # train_model(data['abdominal_data'], data['chest_data'], data['timestamp'])
        return jsonify({
            "timestamp_data": "TBD",
            "fetal_ecg_data": "TBD",
            "maternal_ecg_data": "TBD"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/load-model', methods=['POST'])
def load_model():
    data = request.get_json()
    if not data or 'id' not in data:
        return jsonify({"error": "Invalid request parameters"}), 400

    try:
        # Placeholder for loading and training pre-trained model logic
        # load_and_train_model(id, data.get('abdominal_data'), data.get('chest_data'), data.get('timestamp'))
        return jsonify({
            "timestamp_data": "TBD",
            "fetal_ecg_data": "TBD",
            "maternal_ecg_data": "TBD"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/separate-ecg', methods=['POST'])
def separate_ecg():
    data = request.get_json()
    is_valid, error_message = validate_data_train(data)
    if not is_valid:
        return jsonify({"error": error_message}), 400

    try:
        # Placeholder for ECG signal separation logic
        # fetal_ecg, maternal_ecg = separate_signals(data['abdominal_data'], data['chest_data'])
        return jsonify({
            "timestamp_data": "TBD",
            "fetal_ecg_data": "TBD",
            "maternal_ecg_data": "TBD"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
