from flask import Blueprint, request
import orjson
import os
from ..utils.validation import validate_payload
from ..utils.file_operations import load_users, save_users, get_base_model

bp = Blueprint("user_routes", __name__)


@bp.route("/user/<id>", methods=["GET"])
def get_user(id):
    try:
        users_data = load_users()
        user_data = next(
            (user for user in users_data["users"] if user["id"] == id), None
        )

        if user_data is None:
            return orjson.dumps({"error": "User not found"}), 404

        return orjson.dumps({"user_data": user_data}), 200
    except Exception as e:
        return orjson.dumps({"error": str(e)}), 500


@bp.route("/user/<id>", methods=["POST"])
def create_user(id):
    try:
        data = request.get_json()
        if not data:
            return orjson.dumps({"error": "Missing JSON payload"}), 400

        is_valid, result = validate_payload(data, id)
        if not is_valid:
            return orjson.dumps({"error": result}), 400

        users_data = load_users()

        user_data = next(
            (user for user in users_data["users"] if user["id"] == id), None
        )

        if user_data is not None:
            return orjson.dumps({"error": "User already exists"}), 409

        base_model_path = get_base_model()
        if base_model_path:
            result["models"] = [base_model_path]
        else:
            result["models"] = []
        users_data["users"].append(result)
        save_users(users_data)
        return orjson.dumps({"success": "User created successfully"}), 200

    except Exception as e:
        return orjson.dumps({"error": str(e)}), 500


@bp.route("/user/<id>", methods=["PUT"])
def update_user(id):
    try:
        data = request.get_json()
        if not data:
            return orjson.dumps({"error": "Missing JSON payload"}), 400

        is_valid, result = validate_payload(data, id)
        if not is_valid:
            return orjson.dumps({"error": result}), 400

        users_data = load_users()

        user_data = next(
            (user for user in users_data["users"] if user["id"] == id), None
        )

        if user_data is None:
            return orjson.dumps({"error": "User not found"}), 404

        user_data.update(result)
        save_users(users_data)
        return orjson.dumps({"success": "User updated successfully"}), 200

    except Exception as e:
        return orjson.dumps({"error": str(e)}), 500


@bp.route("/user/<id>", methods=["DELETE"])
def delete_user(id):
    try:
        users_data = load_users()
        user_data = next(
            (user for user in users_data["users"] if user["id"] == id), None
        )

        if user_data is None:
            return orjson.dumps({"error": "User not found"}), 404

        users_data["users"] = [user for user in users_data["users"] if user["id"] != id]
        save_users(users_data)

        # Delete the corresponding model data
        model_path = user_data.get("model_path")
        if model_path != "" and os.path.exists(model_path):
            os.remove(model_path)

        return orjson.dumps({"success": "User deleted successfully"}), 200
    except Exception as e:
        return orjson.dumps({"error": str(e)}), 500


@bp.route("/user_models/<id>", methods=["GET"])
def get_all_models_names(id):
    try:
        users_data = load_users()
        user_data = next(
            (user for user in users_data["users"] if user["id"] == id), None
        )

        if user_data is None:
            return orjson.dumps({"error": "User not found"}), 404

        return orjson.dumps({"models": user_data.get("models")}), 200
    except Exception as e:
        return orjson.dumps({"error": str(e)}), 500
