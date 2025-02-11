#!/usr/bin/env python

import subprocess
import tkinter as tk
from tkinter import messagebox
import os


# Function to check if NVIDIA GPU with CUDA support is available
def check_cuda_gpu_available():
    try:
        # Run nvidia-smi to check for GPU availability
        subprocess.check_call(['nvidia-smi'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True  # CUDA-compatible GPU is available
    except subprocess.CalledProcessError:
        return False  # No CUDA-compatible GPU found


def check_docker_installed():
    try:
        subprocess.check_call(['docker', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def check_docker_running():
    try:
        subprocess.check_call(['docker', 'info'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def check_docker_compose_installed():
    try:
        subprocess.check_call(['docker-compose', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def show_error_message(message):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    messagebox.showerror("Error", message)
    root.destroy()


def run_docker():
    if not check_docker_installed():
        show_error_message(
            "Docker is not installed. Please install Docker from https://www.docker.com/products/docker-desktop and try again.")
        return

    if not check_docker_running():
        show_error_message("Docker is not running. Please start Docker and try again.")
        return

    if not check_docker_compose_installed():
        show_error_message(
            "Docker Compose is not installed. Please install Docker Compose from https://docs.docker.com/compose/ and try again.")
        return

    if not os.path.exists('docker-compose.yml'):
        show_error_message(
            "No docker-compose.yml file found. Please ensure you're in the correct directory with the Docker Compose file.")
        return

    # Check if CUDA-supported GPU is available
    if check_cuda_gpu_available():
        print("CUDA-compatible GPU detected! Running Docker container with GPU support...")
        try:
            env = os.environ.copy()
            env['RUNTIME'] = 'nvidia'
            # Build and run the Docker container with GPU support, passing USE_GPU=true
            subprocess.check_call(['docker-compose', 'build', '--build-arg', 'USE_GPU=true'])
            subprocess.check_call(['docker-compose', 'up', '--remove-orphans'])
        except subprocess.CalledProcessError as e:
            show_error_message(f"An error occurred: {e}")
    else:
        print("No CUDA-compatible GPU detected. Running Docker container in CPU-only mode...")
        try:
            # Build and run the Docker container in CPU-only mode, passing USE_GPU=false
            subprocess.check_call(['docker-compose', 'build', '--build-arg', 'USE_GPU=false'])
            subprocess.check_call(['docker-compose', 'up', '--remove-orphans'])
        except subprocess.CalledProcessError as e:
            show_error_message(f"An error occurred: {e}")


if __name__ == "__main__":
    run_docker()
