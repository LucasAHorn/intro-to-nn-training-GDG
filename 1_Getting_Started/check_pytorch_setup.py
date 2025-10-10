# check_pytorch_setup.py
"""
Check PyTorch and CUDA environment.

This script reports:
- Installed PyTorch version
- CPU functionality
- GPU functionality (if available)
- CUDA device information
"""

import torch

print("PyTorch Environment Diagnostic")
print("----------------------------------------")

# PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check CPU functionality
try:
    cpu_tensor = torch.rand(2, 2)
    print("CPU test: PASS (CPU tensor created successfully)")
except Exception as e:
    print(f"CPU test: FAIL ({e})")

# Check GPU availability and functionality
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    try:
        gpu_tensor = torch.rand(2, 2).cuda()
        print("GPU test: PASS (GPU tensor created successfully)")
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices detected: {num_devices}")
        for i in range(num_devices):
            print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"GPU test: FAIL ({e})")
        # Still list available devices if possible
        num_devices = torch.cuda.device_count()
        if num_devices > 0:
            print(f"Number of CUDA devices detected: {num_devices}")
            for i in range(num_devices):
                print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("No CUDA devices detected.")
else:
    print("GPU test: SKIPPED (CUDA not available)")

print("----------------------------------------")
print("Environment check complete.")
