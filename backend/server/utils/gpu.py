import torch
import time
import threading
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = torch.empty((1, 2, 1024), dtype=torch.float32, device=DEVICE)


def keep_gpu_warm(global_model):
    while True:
        try:
            if global_model is not None:
                with torch.no_grad():
                    _ = global_model(torch.randn(1, 2, 1024, dtype=torch.float32, device=DEVICE))
            time.sleep(0.05)
        except Exception as e:
            logging.error(f"GPU warm-up thread error: {e}")


def start_gpu_warmup(global_model):
    if global_model is None:
        logging.error("Global model is not defined.")
        return
    warmup_thread = threading.Thread(target=keep_gpu_warm, args=(global_model,), daemon=True)
    warmup_thread.start()
