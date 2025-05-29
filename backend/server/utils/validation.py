import time
from ..ecg_data_pb2 import AbdominalData, ChestData


def is_valid_protobuf(data, field, protobuf_class):
    if field in data:
        if not isinstance(data[field], protobuf_class):
            return False, f"{field} must be a {protobuf_class.__name__} instance"
    return True, ""


def validate_data_train(data):
    missing_fields = [
        field
        for field in ["abdominal_data", "chest_data", "timestamp"]
        if field not in data
    ]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"

    for field, protobuf_class in [
        ("abdominal_data", AbdominalData),
        ("chest_data", ChestData),
    ]:
        is_valid, error_message = is_valid_protobuf(data, field, protobuf_class)
        if not is_valid:
            return False, error_message

    try:
        time.strptime(data["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return (
            False,
            "Invalid timestamp format, expected ISO 8601 (e.g., '2024-10-27T12:00:00Z')",
        )

    return True, ""


def validate_payload(data, id):
    required_fields = {
        "name",
        "maternal_age",
        "gestational_age",
        "clinician_name",
        "referral_reason",
        "additional_comments",
    }
    if not isinstance(data, dict) or not required_fields.issubset(data.keys()):
        return False, "Invalid or missing fields in request payload."

    if not isinstance(data["name"], str):
        return False, "Name must be strings."

    try:
        data["maternal_age"] = float(data["maternal_age"])
    except ValueError:
        return False, "Maternal age must be a float."

    data["models"] = []
    data["id"] = id

    return True, data
