import requests
from ecg_data_pb2 import AbdominalData, ChestData, CapturedECGData

# URL of the Flask endpoint
url = "http://localhost:5000/separate-ecg"
model_url = "http://localhost:5000/load-model"  # URL to load the model
v2_url = "http://localhost:5000/v2-separate-ecg"


# Function to generate and validate protobuf data
def generate_protobuf_data(iterations=5):
    for i in range(iterations):  # Simulate sending 5 protobuf message
        # Example data for abdominal and chest
        abdominal_values = [0.1 * j for j in range(1024)]  # Example abdominal data
        chest_values = [0.2 * j for j in range(1024)]  # Example chest data

        # Ensure all required fields are populated
        if len(abdominal_values) == 1024 and len(chest_values) == 1024:
            # Create AbdominalData and ChestData messages
            abdominal_data = AbdominalData(values=abdominal_values)
            chest_data = ChestData(values=chest_values)

            # Create CapturedECGData message
            ecg_data = CapturedECGData(
                abdominal_data=abdominal_data,
                chest_data=chest_data,
                timestamp=f"2024-11-25T12:00:0{i}Z"
            )

            # Serialize the message and yield it
            yield ecg_data.SerializeToString()
        else:
            print("Incomplete data; skipping this iteration.")


# Function to load the model
def load_model():
    try:
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "id": "12345"
        }
        response = requests.post(model_url, headers=headers, json=data)
        if response.status_code == 200:
            print("Model loaded successfully.")
            return True
        else:
            print(f"Failed to load model. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"An error occurred while loading the model: {str(e)}")
        return False


def test_separate_ecg_v2():
    # Create a generator for protobuf data
    data_generator = generate_protobuf_data(1)

    # Send the protobuf data to the endpoint
    try:
        response = requests.post(url, data=data_generator)
        if response.status_code == 200:
            print("Response received:")
            print(response.json())
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    # First, try to load the model
    if not load_model():
        print("Model loading failed. Aborting ECG processing.")
    test_separate_ecg_v2()
    test_separate_ecg_v2()
    test_separate_ecg_v2()
    test_separate_ecg_v2()
    test_separate_ecg_v2()
