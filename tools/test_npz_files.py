import numpy as np

# Load an NPZ file and check its keys
data = np.load('tools/validation_output/session_20260113_174235/good/frame_000192.npz')
print("Available keys:", list(data.keys()))
print("Data shapes:", {k: v.shape for k, v in data.items()})
