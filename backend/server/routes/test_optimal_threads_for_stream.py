import logging
import time
import torch
import threading
import random
import sys
import os
import numpy as np

sys.path.append('/app/server')

# from ResnetNetwork import *  # for local testing
# from ecg_data_pb2 import AbdominalData, ChestData, CapturedECGData  # for local testing
from ..ResnetNetwork import *  # for docker
from ..ecg_data_pb2 import AbdominalData, ChestData, CapturedECGData  # for docker

# Global variable for the model
global_model = None

MAX_WORKERS = 10
EXPECTED_SIZE = 8226  # Size threshold for a complete Protobuf message

# Simulate loading a pre-trained model
def load_model():
    global global_model
    try:
        # model_path = f"../db/models/base_model_16-11-24.pt"  # Adjust path as needed
        model_path = os.path.join(os.getcwd(), f"../db/models/base_model_16-11-24.pt")  # model.module for docker
        global_model = torch.load(model_path, map_location=torch.device('cpu'))
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")

# Simulate processing a chunk of data
def process_chunk(message_bytes):
    try:
        # Simulate parsing Protobuf message
        ecg_data = CapturedECGData()
        ecg_data.ParseFromString(message_bytes)

        # Dummy processing function for ECG data
        abdominal_data = ecg_data.abdominal_data.values
        chest_data = ecg_data.chest_data.values
        timestamp = ecg_data.timestamp

        # Simulating model inference
        abdominal_tensor = torch.tensor(abdominal_data).view(1, 1, 1024).float()
        chest_tensor = torch.tensor(chest_data).view(1, 1, 1024).float()
        input_tensor = torch.cat((abdominal_tensor, chest_tensor), dim=1)

        with torch.no_grad():
            model_output = global_model(input_tensor)

        return model_output
    except Exception as e:
        logging.error(f"Failed to process chunk: {str(e)}")
        return None

# Function to simulate sending a Protobuf message as chunks
def generate_random_chunk():
    abdominal_data = [random.random() for _ in range(1024)]
    chest_data = [random.random() for _ in range(1024)]
    timestamp = "2024-10-27T12:00:00Z"
    ecg_sample = CapturedECGData(
        abdominal_data=AbdominalData(values=abdominal_data),
        chest_data=ChestData(values=chest_data),
        timestamp=timestamp
    )
    return ecg_sample.SerializeToString()

# Worker function for each thread
def worker_thread(chunk, results, index):
    results[index] = process_chunk(chunk)

# Function to test optimal number of threads
def test_optimal_threads():
    load_model()

    chunks = [generate_random_chunk() for _ in range(10)]  # 10 chunks for testing
    results = [None] * len(chunks)

    best_time = float('inf')
    best_threads = 0
    best_time_per_thread = float('inf')

    # Test for different number of threads
    for num_threads in range(1, MAX_WORKERS + 1):
        start_time = time.time()

        # Create threads for processing chunks
        threads = []
        for i in range(num_threads):
            chunk_index = i % len(chunks)  # Distribute chunks to threads
            thread = threading.Thread(target=worker_thread, args=(chunks[chunk_index], results, chunk_index))
            threads.append(thread)

        # Start and join threads
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_thread = elapsed_time / num_threads  # Calculate time per thread

        logging.info(f"Processing time with {num_threads} threads: {elapsed_time:.6f} seconds")
        logging.info(f"Time per thread with {num_threads} threads: {time_per_thread:.6f} seconds")

        # Check if this configuration is the best so far (based on time per thread)
        if time_per_thread < best_time_per_thread:
            best_time_per_thread = time_per_thread
            best_threads = num_threads
            best_time = elapsed_time  # Update best total time as well

    logging.info(f"Optimal number of threads: {best_threads} with processing time: {best_time:.6f} seconds")
    logging.info(f"Best time per thread: {best_time_per_thread:.6f} seconds")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_optimal_threads()
