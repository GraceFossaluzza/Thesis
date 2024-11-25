import numpy as np
import pandas as pd

channel = 35
data = pd.read_csv("eeg_data.csv")
print("shape of data: ", data.shape)

# Select the data for the specific channel and convert it to a NumPy array
eeg_data = data.iloc[channel, :].to_numpy()
n = data.shape[1]
#n_epochs = 1000
#time_steps_epoch = n / n_epochs

# Parameters for overlapping windows
window_size = 15  # Length of each window
overlap = 10    # Overlap between windows

# Function to create overlapping windows
def overlapping_windows(arr, window_size, overlap):
    step = window_size - overlap  # Calculate the step size
    windows = []
    
    for start in range(0, len(arr) - window_size + 1, step):
        end = start + window_size
        windows.append(arr[start:end])
    
    return windows

# Split the eeg_data into overlapping windows
windows = overlapping_windows(eeg_data, window_size, overlap)

# Convert the windows list to a NumPy array
windows_array = np.array(windows)

# Save the array to a .npy file
np.save("eeg_windows.npy", windows_array)

# Optionally, print the saved windows to verify
print(f"Saved windows to 'eeg_windows.npy' with shape: {windows_array.shape}")
