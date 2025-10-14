# Getting Started with PyTorch

This repository provides a simple setup guide to get started using **PyTorch** for neural network projects.  
It includes:
- Environment setup instructions  
- Optional CUDA (GPU) configuration for NVIDIA users  
- A script to install all required dependencies  
- A test script to confirm PyTorch is working correctly  

---

## 1. Getting Started

Before starting, make sure you have:
- **Python 3.8+**
- **pip** (Python package manager)
- (Optional) NVIDIA CUDA Toolkit — only if you have an NVIDIA GPU and plan to use it for acceleration

You can verify your Python and pip versions:
```bash
python --version
pip --version
```

## 2. Function Approximation
- This has a file that will train a model to approximate the taylor series
- Users will need to fill in the blanks with information before it is runnable

## 3. Weather Prediction
- This has a completed py file that will train a model on a small set of data and save the model
- Also contains a py file to test the model on a small set of data
- The data provided is hourly weather information with the future temp included, for 500 hours and for 1 year

## Completed Models
- This contains 'answers' to parts 2 and 3
    - approxomates sin from -pi to pi
    - fully functional nn trainer on the year worth of data