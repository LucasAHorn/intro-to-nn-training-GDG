# install_pytorch_packages.py
import os

packages = [
    "torch",
    "numpy",
    "matplotlib",
]

os.system(f"pip install {' '.join(packages)}")
print("\tBasic PyTorch setup complete.")