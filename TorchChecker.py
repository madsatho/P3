import subprocess
import sys

try:
    import torch
except ModuleNotFoundError:
    # Installs PyTorch (CUDA 13.2 build) into the same Python that runs this script,
    # so it can't end up in a different interpreter than PyCharm uses.
    INDEX_URL = "https://download.pytorch.org/whl/cu126"

    print("Installing into:", sys.executable)

    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision",
        "--index-url", INDEX_URL,
    ])
    print("Pytorch has been installed")

finally:
    # Verify the install
    import torch

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))