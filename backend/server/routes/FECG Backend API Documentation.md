# User Management API Documentation

## Overview

This API provides endpoints for managing user data, including retrieving, creating, updating, and deleting user records.
All data is stored in a JSON file and operations are performed through RESTful endpoints.

## Base URL

All endpoints are relative to the base URL of your server.

## Authentication

Authentication requirements are not specified in the provided code. Please refer to your server's authentication
documentation.

## Endpoints

### Get User Information

Retrieves information for a specific user.

```
GET /user/{id}
```

#### Path Parameters

| Parameter | Type   | Description    |
|-----------|--------|----------------|
| id        | string | Unique user ID |

#### Responses

| Status Code | Description  | Response Body                 |
|-------------|--------------|-------------------------------|
| 200         | Success      | `{"user_data": {...}}`        |
| 404         | Not Found    | `{"error": "User not found"}` |
| 500         | Server Error | `{"error": "Error message"}`  |

#### Example

```
GET /user/12345
```

Response:

```json
{
  "user_data": {
    "name": "John Doe",
    "maternal_age": 28.5,
    "gestational_age": 30,
    "clinician_name": "Dr. Smith",
    "referral_reason": "Routine check-up",
    "additional_comments": "No complications observed."
  }
}

```

### Create or Update User

Creates a new user or updates an existing user's information.

```
POST /user/{id}
```

#### Path Parameters

| Parameter | Type   | Description    |
|-----------|--------|----------------|
| id        | string | Unique user ID |

#### Request Body

JSON object containing user data. The specific fields required are handled by the `validate_payload` function.

#### Responses

| Status Code | Description  | Response Body                               |
|-------------|--------------|---------------------------------------------|
| 200         | Updated      | `{"success": "User updated successfully."}` |
| 201         | Created      | `{"success": "User added successfully."}`   |
| 400         | Bad Request  | `{"error": "Validation error message"}`     |
| 500         | Server Error | `{"error": "Error message"}`                |

#### Example

```
POST /user/12345
Content-Type: application/json

{
  "user_data": {
    "id": "12345",
    "name": "John Doe",
    "maternal_age": 28.5,
    "gestational_age": 30,
    "clinician_name": "Dr. Smith",
    "referral_reason": "Routine check-up",
    "additional_comments": "No complications observed.",
    "model_path": ""
  }
}

```

Response:

```json
{
  "success": "User updated successfully."
}
```

### Delete User

Deletes a user and their associated model data.

```
DELETE /user/{id}
```

#### Path Parameters

| Parameter | Type   | Description    |
|-----------|--------|----------------|
| id        | string | Unique user ID |

#### Responses

| Status Code | Description  | Response Body                               |
|-------------|--------------|---------------------------------------------|
| 200         | Success      | `{"success": "User deleted successfully."}` |
| 404         | Not Found    | `{"error": "User not found"}`               |
| 500         | Server Error | `{"error": "Error message"}`                |

#### Example

```
DELETE /user/12345
```

Response:

```json
{
  "success": "User deleted successfully."
}
```

### List All Models for User

Retrieves a list of all available model files for a specific user based on his id.

```
GET /user_models/<id>
```

#### Responses

| Status Code | Description  | Response Body                                 |
|-------------|--------------|-----------------------------------------------|
| 200         | Success      | `{"models": ["model1.pt", "model2.pt", ...]}` |
| 404         | Not Found    | `{"error": "User not found"}`                 |
| 500         | Server Error | `{"error": "Error"}`                          |

#### Example

```
GET /user_models/12345
```

Response:

```json
{
  "models": [
    "base_model_16-11-24.pt",
    "12345-model.pt"
  ]
}
```

### Get Protobuf Message Size

Retrieves the size in bytes of a sample ECG protobuf message. This is useful for client applications to understand
memory requirements.

```
GET /size
```

#### Responses

| Status Code | Description  | Response Body                |
|-------------|--------------|------------------------------|
| 200         | Success      | `{"message_size": 8192}`     |
| 500         | Server Error | `{"error": "Error message"}` |

#### Example

```
GET /size
```

Response:

```json
{
  "message_size": 8192
}
```

### Load Model

Loads a specific model into memory for ECG processing.

```
POST /load-model
```

#### Request Body

| Field | Type   | Description      | Required |
|-------|--------|------------------|----------|
| id    | string | Model identifier | Yes      |

#### Responses

| Status Code | Description  | Response Body                                                                            |
|-------------|--------------|------------------------------------------------------------------------------------------|
| 200         | Success      | `{"status": "success", "message": "Model loaded successfully.", "model_id": "model_id"}` |
| 400         | Bad Request  | `{"error": "ID doesnt match a model"}`                                                   |
| 500         | Server Error | `{"error": "Error message"}`                                                             |

#### Notes

- If the specified model is not found, the system will fall back to the default `base_model_16-11-24.pt`.
- The endpoint starts a GPU warm-up process after loading the model.

#### Example

```
POST /load-model
Content-Type: application/json

{
  "id": "custom_model_v2"
}
```

Response:

```json
{
  "status": "success",
  "message": "Model loaded successfully.",
  "model_id": "custom_model_v2"
}
```

### Separate ECG

Processes ECG data and separates components using the currently loaded model.

```
POST /separate-ecg
```

#### Request Body

Binary Protobuf message containing the ECG data in the `CapturedECGData` format with:

- `abdominal_data`: Array of 1024 float values
- `chest_data`: Array of 1024 float values
- `timestamp`: ISO-8601 formatted timestamp

#### Responses

| Status Code | Description  | Response Body                                                      |
|-------------|--------------|--------------------------------------------------------------------|
| 200         | Success      | JSON object containing separated ECG data (see example below)      |
| 400         | Bad Request  | `{"error": "Validation error message"}`                            |
| 500         | Server Error | `{"error": "Model is not loaded"}` or `{"error": "Error message"}` |

#### Successful Response Format

```json
{
  "timestamp_data": "2024-10-27T12:00:00Z",
  "fetal_ecg_data": [[...]],
  "tensor2_data": [[...]],
  "maternal_ecg_data": [[...]],
  "tensor4_data": [[...]]
}
```

#### Response Fields

| Field             | Type   | Description                             |
|-------------------|--------|-----------------------------------------|
| timestamp_data    | string | Original timestamp from the input data  |
| fetal_ecg_data    | array  | Separated fetal ECG signal data         |
| tensor2_data      | array  | Additional tensor output from the model |
| maternal_ecg_data | array  | Separated maternal ECG signal data      |
| tensor4_data      | array  | Additional tensor output from the model |

#### Notes

- A model must be loaded via the `/load-model` endpoint before using this endpoint.
- The endpoint measures and logs processing time.
- The request must be a valid Protobuf message in the expected format.
- Input data arrays must contain exactly 1024 elements each.

#### Example

```
POST /separate-ecg
Content-Type: application/x-protobuf

[Binary Protobuf data]
```

## User Data Model

Users are stored in a JSON structure. The exact schema is not specified in the provided code, but it includes at least:

- `id`: Unique identifier for the user
- `model_path`: Optional path to a model file associated with the user

## Error Handling

All endpoints return appropriate HTTP status codes:

- 200 for successful operations
- 201 for successful creation
- 400 for client errors (invalid input)
- 404 when a requested resource is not found
- 500 for server errors

Error responses include a JSON object with an `error` field containing a message describing the error.

### Protobuf Schema

The API expects ECG data in a protobuf format with the following structure:

```
message CapturedECGData {
  AbdominalData abdominal_data = 1;
  ChestData chest_data = 2;
  string timestamp = 3;
}

message AbdominalData {
  repeated float values = 1;  // Expected to have 1024 values
}

message ChestData {
  repeated float values = 1;  // Expected to have 1024 values
}
```

## Dependencies

- The API requires access to models in the `MODEL_DIR` directory.
- A GPU may be required for optimal performance.
- Models are expected to be PyTorch models in `.pt` format.
