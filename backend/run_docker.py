#!/usr/bin/env python

import subprocess
import tkinter as tk
from tkinter import messagebox


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


def show_error_message(message):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    messagebox.showerror("Error", message)
    root.destroy()


def run_docker():
    if not check_docker_installed():
        show_error_message("Docker is not installed. Please install Docker from https://www.docker.com/products/docker-desktop and try again.")
        return

    if not check_docker_running():
        show_error_message("Docker is not running. Please start Docker and try again.")
        return

    try:
        # Build the Docker image
        subprocess.check_call(['docker-compose', 'build'])
        # Run the Docker container with --remove-orphans flag
        subprocess.check_call(['docker-compose', 'up', '--remove-orphans'])
    except subprocess.CalledProcessError as e:
        show_error_message(f"An error occurred: {e}")


if __name__ == "__main__":
    run_docker()
