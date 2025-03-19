import os
import orjson

USER_FILE_PATH = os.path.join(os.getcwd(), "server/db/users/users.json")


def load_users():
    if not os.path.exists(USER_FILE_PATH):
        raise FileNotFoundError(f"User file not found at {USER_FILE_PATH}")

    with open(USER_FILE_PATH, 'r', encoding='utf-8') as file:
        return orjson.loads(file.read())


def save_users(users_data):
    with open(USER_FILE_PATH, 'w', encoding='utf-8') as file:
        file.write(orjson.dumps(users_data).decode('utf-8'))
