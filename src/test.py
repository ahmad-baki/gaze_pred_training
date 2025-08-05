import numpy as np

# loads numpy-array of gaze vectors
GAZE_VECTORS_FILE = '/pfs/work9/workspace/scratch/ka_eb5961-holo2gaze/old_frame/processed/3d/pear_banana_in_sink/2025_07_03-13_09_20/sensors/continuous_device_/gaze_vectors.npy'
gaze_vectors = np.load(GAZE_VECTORS_FILE)
print(f"Loaded gaze vectors with shape: {gaze_vectors.shape}")
print(f"First gaze vector: {gaze_vectors[0]}")
