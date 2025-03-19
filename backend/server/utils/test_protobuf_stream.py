# # Send the protobuf data to the endpoint with stream
# def test_separate_ecg():
#     # First, try to load the model
#     if not load_model():
#         print("Model loading failed. Aborting ECG processing.")
#         return
#
#     headers = {
#         "Content-Type": "application/octet-stream",
#     }
#
#     # Create a generator for protobuf data
#     data_generator = generate_protobuf_data(5)
#
#     try:
#         with requests.post(url, data=data_generator, headers=headers, stream=True) as response:
#             # Check the status of the response
#             if response.status_code == 200:
#                 print("Response received:")
#                 # Iterate over each line in the response stream
#                 for line in response.iter_lines():
#                     if line:
#                         # Decode each line from bytes to a string
#                         line = line.decode('utf-8')
#
#                         # Check if the line starts with "data: ", which is the SSE format
#                         if line.startswith('data: '):
#                             try:
#                                 # Parse the JSON data (strip 'data: ' and parse the rest)
#                                 data = json.loads(line[6:])  # Skip 'data: ' part
#                                 print("Received data:", data)
#                             except json.JSONDecodeError:
#                                 print("Error parsing JSON:", line)
#             else:
#                 print(f"Error {response.status_code}: {response.text}")
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")