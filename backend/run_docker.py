import subprocess
import tkinter as tk
from tkinter import messagebox
import os
import sys
import signal
import atexit


# Utility function to check if a command can be run successfully
def check_dependency(command, error_message):
    try:
        subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Only show error for non-nvidia-smi commands
        if command[0] != 'nvidia-smi':
            show_error_message(error_message)
        return False


# Function to check if CUDA GPU is available
def check_cuda_gpu_available():
    return check_dependency(['nvidia-smi'], "")


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


# Global process variable to track the docker-compose process
docker_process = None


# Function to handle graceful shutdown
def graceful_shutdown(signum=None, frame=None):
    if docker_process and docker_process.poll() is None:
        print("\nShutting down Docker containers gracefully...")
        try:
            # Use docker-compose down to gracefully stop containers
            compose_file = docker_process.args[2]
            subprocess.run(['docker-compose', '-f', compose_file, 'down'],
                           check=False, timeout=30)
        except (subprocess.SubprocessError, FileNotFoundError):
            # If docker-compose down fails, kill the process directly
            print("Forcefully terminating Docker process...")
            docker_process.terminate()
            docker_process.wait(timeout=5)

    # Exit without error
    sys.exit(0)


def run_docker():
    global docker_process

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    atexit.register(graceful_shutdown)

    # Check if dependencies are installed and running
    checks = [
        (['docker', '--version'], "Docker is not installed. Please install Docker and try again."),
        (['docker', 'info'], "Docker is not running. Please start Docker and try again."),
        (['docker-compose', '--version'],
         "Docker Compose is not installed. Please install Docker Compose and try again.")
    ]

    for cmd, err_msg in checks:
        if not check_dependency(cmd, err_msg):
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
        docker_compose_path = get_resource_path('docker-compose.gpu.yml')
        use_gpu = 'true'
    else:
        print("No CUDA-compatible GPU detected. Running Docker container in CPU-only mode...")
        docker_compose_path = get_resource_path('docker-compose.cpu.yml')
        use_gpu = 'false'

    if not os.path.exists(docker_compose_path):
        # Try to find any docker-compose file as fallback
        fallback_files = ['docker-compose.yml', 'docker-compose.yaml']
        for file in fallback_files:
            path = get_resource_path(file)
            if os.path.exists(path):
                docker_compose_path = path
                break
        else:
            show_error_message(
                f"No docker-compose file found. Please ensure you're in the correct directory with the Docker Compose file.")
            return

    # Build and run the Docker container with the appropriate GPU settings
    try:
        # Run docker-compose build
        subprocess.check_call(
            ['docker-compose', '-f', docker_compose_path, 'build', '--build-arg', f'USE_GPU={use_gpu}'],
            env=env)

        # Run docker-compose up in a way we can handle gracefully
        docker_process = subprocess.Popen(
            ['docker-compose', '-f', docker_compose_path, 'up', '--remove-orphans'],
            env=env)

        # Wait for the process to complete
        docker_process.wait()

        # Check the return code
        if docker_process.returncode != 0 and docker_process.returncode != 130:  # 130 is the return code for SIGINT
            show_error_message(
                f"Docker exited with code {docker_process.returncode}. Please check the logs for details.")

    except subprocess.CalledProcessError as e:
        # Only show error message for build errors or other non-termination errors
        if e.returncode != 130:  # 130 is the return code for SIGINT
            error_message = f"An error occurred while running Docker: {e}\n"
            error_message += f"Return code: {e.returncode}\n"

            # Optionally, capture stderr if available
            if hasattr(e, 'stderr') and e.stderr:
                error_message += f"stderr: {e.stderr.decode('utf-8')}"

            # Display the error message in a dialog box
            show_error_message(error_message)

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        graceful_shutdown()


if __name__ == "__main__":
    run_docker()
