#!/usr/bin/env python3
import os

from rosa_compute import inspect_checkpoint, print_diagnostics

if __name__ == "__main__":
    print_diagnostics()
    model_path = os.environ.get("ROSA_MODEL_PATH")
    if model_path:
        print("\n=== ROSA Model Checkpoint Inspection ===")
        if os.path.exists(model_path):
            info = inspect_checkpoint(model_path)
            print(f"Path:             {info['checkpoint_path']}")
            print(f"File Size:        {info['file_size_mb']:.2f} MB")
            print(f"SHA-256:          {info['sha256']}")
            print(f"Num Tensors:      {info['num_tensors']}")
            print(f"Total Parameters: {info['total_parameters']:,}")
        else:
            print(f"ROSA_MODEL_PATH set to '{model_path}', but file does not exist.")
