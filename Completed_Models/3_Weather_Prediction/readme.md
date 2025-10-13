# Temperature Prediction: Yearly Data Model

## Overview

This project demonstrates how to predict temperatures using a full year of hourly/daily data (`Data/future_temp_year.csv`) with a more advanced neural network.  

If you’ve tried the simpler 500-line temperature prediction model, this project extends it by:

- Using **more layers** in the neural network  
- Implementing **dropout** to reduce overfitting  
- Using **adaptive learning rates** with a scheduler
- Applying **early stopping** to prevent unnecessary training

The goal is to predict the temperature based on multiple input features and learn patterns from a large dataset.


---

## `Network_Creation.py` - More Information

1. **Load CSV Data**  
   Reads `future_temp_year.csv` and converts it to a NumPy array, then splits it into:

   - `X` → Features (inputs)  
   - `y` → Target (temperature to predict)

2. **Normalize Data**  
   Inputs and outputs are normalized (mean 0, std 1) to help the neural network train efficiently.  
   > ⚡ Important: Normalization ensures all features have similar scale, which prevents the network from favoring certain inputs.

3. **Dataset and DataLoader**  
   - Splits data into **training** and **validation** sets (90/10 split)  
   - Uses PyTorch `DataLoader` for batching during training

4. **Define Neural Network**  
   - Input layer → Hidden layers (64 → 32 → 16 units) → Output layer  
   - Uses **ReLU activation** for non-linearity  
   - **Dropout (10%)** prevents overfitting

5. **Training Setup**  
   - **Loss function:** Smooth L1 Loss (more robust than MSE)  
   - **Optimizer:** Adam  
   - **Learning rate scheduler:** reduces LR when validation loss stops improving  
   - **Early stopping:** stops training if validation loss doesn’t improve for 30 epochs

6. **Training Loop**  
   - Runs for up to 200 epochs  
   - Tracks training and validation loss  
   - Saves the best model using **TorchScript**

---

## How to Run

1. Make sure you have Python 3.10+ and PyTorch installed.
2. Run the script:

```bash
python Completed_Models/3_Weather_Prediction/Network_Creation.py
