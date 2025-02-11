# For working with stream data
# @bp.route('/separate-ecg', methods=['POST'])
# def separate_ecg():
#     start_time = time.time()  # Start timer
#     global model
#
#     if model is None:
#         return orjson.dumps({"error": "Model is not loaded"}), 500
#
#     def stream_responses():
#         buffer = b''  # Initialize an empty buffer to accumulate chunks
#
#         for chunk in request.stream:
#             buffer += chunk  # Add chunk to the buffer
#
#             # Process complete messages from the buffer
#             while len(buffer) >= EXPECTED_SIZE:
#                 # Extract a single complete message
#                 message_bytes = buffer[:EXPECTED_SIZE]
#                 buffer = buffer[EXPECTED_SIZE:]  # Retain extra data for the next message
#                 response = process_chunk(message_bytes)  # Process the extracted message
#                 if response[1] != 200:
#                     logging.error(f"Error processing Protobuf message: {response[0]}")
#                 yield f"data: {response[0].get_data(as_text=True)}\n\n"
#
#         # Handle any remaining incomplete message in the buffer
#         if len(buffer) > 0:
#             yield f"data: {orjson.dumps({'error': 'Incomplete Protobuf message received'}).get_data(as_text=True)}\n\n"
#
#         # Calculate and log total processing time
#         total_time = time.time() - start_time
#         logging.info(f"Total processing time for /separate-ecg: {total_time:.6f} seconds")
#
#     # Return a stream response using server-sent events
#     return Response(stream_with_context(stream_responses()), content_type='text/event-stream')