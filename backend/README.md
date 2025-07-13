# FECG Backend

A Python-based backend server for Fetal Electrocardiography (FECG) signal processing and analysis using deep learning
models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
    - [For Deployment](#for-deployment)
    - [For Development](#for-development)
- [Usage](#usage)
    - [Key Endpoints](#key-endpoints)
- [Development](#development)
    - [Running Tests](#running-tests)
    - [Building Executables](#building-executables)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview

This backend provides API endpoints for processing ECG data, separating maternal and fetal signals using ResNet-based
neural networks, and managing user data. The system supports both CPU and GPU processing with Docker containerization.

## Features

- **ECG Signal Processing**: Advanced algorithms for maternal-fetal ECG separation
- **Deep Learning Models**: ResNet-based neural networks for signal analysis
- **Protocol Buffers**: Efficient data serialization for ECG streaming
- **User Management**: User authentication and data storage
- **Docker Support**: Containerized deployment with CPU/GPU options
- **API Documentation**: Comprehensive REST API endpoints

## Project Structure

```
backend/
├── server/                # Core server application
│   ├── routes/            # API route handlers
│   ├── utils/             # Utility functions
│   ├── db/                # Database models and user data
│   ├── ResnetNetwork.py   # Neural network implementation
│   └── HelpFunctions.py   # Helper utilities
├── docker-compose.*.yml   # Docker configurations
├── requirements.txt       # Python dependencies
└── dist/                 # Compiled executables
```

## Installation

### Prerequisites

- [Docker Desktop](https://www.docker.com)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yardenhaskin/FECG.git
cd FECG/backend
```

2. Add models directory:

```bash
cd backend/server/db
mkdir models
```

3. Download base model:

   Download the model
   from: [Base Model](https://drive.google.com/file/d/1Flp21LYlwm535sO_T1t5a9zFgEqWxRB8/view?ts=673a96a1)


4. Move the downloaded model to the models' directory:

```bash
mv /path/to/downloaded/model.pth models/
```

### For Deployment

Run the server:

```bash
cd backend/dist
run_FECG_server.exe
```

### For Development

### Prerequisites

- Python 3.10+
- pip package manager
- Docker (optional, for containerized deployment)
- CUDA toolkit (optional, for GPU support)

### Setup

1. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Change imports:

   Make sure to comment relevant imports based on your environment. For example, if you are running the server directly,
   comment out the Docker-related imports in `ResnetBasics.py`, `ResnetNetwork.py`, `routes.py`  and uncomment the
   direct server import in `__init__.py`.


4. Run the server using Docker:

```bash
python run_docker.py
```

5. Alternatively, run the server directly:

```bash
cd backend/server
python __init__.py
```

## Usage

The server provides REST API endpoints for ECG data processing. Refer to
`server/routes/FECG Backend API Documentation.md` for detailed API documentation.

### Key Endpoints

- ECG data upload and processing
- Real-time signal streaming via Protocol Buffers
- User management and authentication
- Model inference and results retrieval

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Building Executables

```bash
python build_exe.py
```

## Dependencies

Core dependencies include:

- Deep learning frameworks for neural network processing
- Protocol Buffers for data serialization
- Flask/FastAPI for web server functionality
- PyTorch for ResNet model implementation

See `requirements.txt` for complete dependency list.

## Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Run tests to ensure functionality
4. Submit a pull request

## License

This project is part of the FECG research initiative. Please refer to the repository license for usage terms.