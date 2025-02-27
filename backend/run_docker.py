import subprocess
import tkinter as tk
from tkinter import messagebox
import os
import sys


# Utility function to check if a command can be run successfully
def check_dependency(command, error_message):
    try:
        subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        show_error_message(error_message)
        return False


# Function to check if CUDA GPU is available
def check_cuda_gpu_available():
    return check_dependency(['nvidia-smi'], "CUDA-compatible GPU is not available. Ensure your GPU supports CUDA.")


# Function to display error messages
def show_error_message(message):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    messagebox.showerror("Error", message)
    root.destroy()


# Get the path to the extracted resources in PyInstaller bundle
def get_resource_path(relative_path):
    try:
        # PyInstaller stores resources in _MEIPASS during runtime
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def run_docker():
    # Check if dependencies are installed and running
    if not check_dependency(['docker', '--version'], "Docker is not installed. Please install Docker and try again."):
        return

    if not check_dependency(['docker', 'info'], "Docker is not running. Please start Docker and try again."):
        return

    if not check_dependency(['docker-compose', '--version'],
                            "Docker Compose is not installed. Please install Docker Compose and try again."):
        return

    # Get the docker-compose.yml path
    docker_compose_path = get_resource_path('docker-compose.yml')

    if not os.path.exists(docker_compose_path):
        show_error_message(
            "No docker-compose.yml file found. Please ensure you're in the correct directory with the Docker Compose file.")
        return

    # Set environment variable for volume path based on whether running with PyInstaller or not
    if hasattr(sys, '_MEIPASS'):
        backend_volume_path = os.path.abspath(os.path.join(os.path.abspath("."), ".."))
    else:
        backend_volume_path = "."

    print("The backend volume path is:", backend_volume_path)

    # Check if CUDA-supported GPU is available and set the appropriate arguments
    env = os.environ.copy()
    env['BACKEND_VOLUME_PATH'] = backend_volume_path

    if check_cuda_gpu_available():
        print("CUDA-compatible GPU detected! Running Docker container with GPU support...")
        env['RUNTIME'] = 'nvidia'
        use_gpu = True
    else:
        print("No CUDA-compatible GPU detected. Running Docker container in CPU-only mode...")
        env['RUNTIME'] = 'cpu'
        use_gpu = False

    # Build and run the Docker container with the appropriate GPU settings
    try:
        subprocess.check_call(
            ['docker-compose', '-f', docker_compose_path, 'build', '--build-arg', f'USE_GPU={str(use_gpu).lower()}'],
            env=env)
        subprocess.check_call(['docker-compose', '-f', docker_compose_path, 'up', '--remove-orphans'], env=env)
    except subprocess.CalledProcessError as e:
        show_error_message(f"An error occurred while running Docker: {e}")


if __name__ == "__main__":
    run_docker()
