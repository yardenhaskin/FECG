import requests
import json
from ecg_data_pb2 import AbdominalData, ChestData, CapturedECGData

# URL of the Flask endpoint
url = "http://localhost:5000/separate-ecg"
model_url = "http://localhost:5000/load-model"  # URL to load the model

# Function to generate and validate protobuf data
def generate_protobuf_data():
    for i in range(5):  # Simulate sending 5 protobuf message
        # Example data for abdominal and chest
        abdominal_values = [0.1 * j for j in range(1024)]  # Example abdominal data
        chest_values = [0.2 * j for j in range(1024)]      # Example chest data

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
        response = requests.post(model_url)
        if response.status_code == 200:
            print("Model loaded successfully.")
            return True
        else:
            print(f"Failed to load model. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"An error occurred while loading the model: {str(e)}")
        return False

# Send the protobuf data to the endpoint
def test_separate_ecg():
    headers = {
        "Content-Type": "application/octet-stream",
    }

    # First, try to load the model
    if not load_model():
        print("Model loading failed. Aborting ECG processing.")
        return

    # Create a generator for protobuf data
    data_generator = generate_protobuf_data()

    try:
        with requests.post(url, data=data_generator, headers=headers, stream=True) as response:
            # Check the status of the response
            if response.status_code == 200:
                print("Response received:")
                # Iterate over each line in the response stream
                for line in response.iter_lines():
                    if line:
                        # Decode each line from bytes to a string
                        line = line.decode('utf-8')

                        # Check if the line starts with "data: ", which is the SSE format
                        if line.startswith('data: '):
                            try:
                                # Parse the JSON data (strip 'data: ' and parse the rest)
                                data = json.loads(line[6:])  # Skip 'data: ' part
                                print("Received data:", data)
                            except json.JSONDecodeError:
                                print("Error parsing JSON:", line)
            else:
                print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_separate_ecg()
