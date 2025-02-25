# Stage 1: Base image
FROM python:3.10-slim AS base

# Set the working directory
WORKDIR /app

# Set build argument to specify whether to use GPU or CPU
ARG USE_GPU=false

# Upgrade pip first to ensure compatibility with package installations
RUN pip install --upgrade pip

# Conditionally copy the appropriate requirements file
COPY requirements-${USE_GPU}.txt /app/requirements-${USE_GPU}.txt
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements-${USE_GPU}.txt

# Expose port for the application (optional in base stage)
EXPOSE 5000

# Stage 2: Runtime image
FROM python:3.10-slim AS runtime

# Set the working directory
WORKDIR /app

# Copy the necessary files, including the installed site packages from the base image
COPY --from=base /app /app
COPY --from=base /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Clean up unnecessary build dependencies (adjust according to what you installed in the base image)
# Remove unnecessary files and packages using apt commands
RUN rm -rf /root/.cache \
    && apt-get update \
    && apt-get remove -y gcc libffi-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Expose the port for the application
EXPOSE 5000

# Define the command to run the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]


