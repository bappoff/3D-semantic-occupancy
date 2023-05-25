import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the .pkl file
with open('/work/scitas-share/voxformer/VoxFormer/3D-semantic-occupancy/results/test.pkl', 'rb') as file:
    data = pickle.load(file)

# Extract the arrays
y_pred_list = [d['y_pred'] for d in data]
y_true_list = [d['y_true'] for d in data]

# Visualize the arrays
fig, axes = plt.subplots(len(y_pred_list), 2, figsize=(10, 10))

for i, (y_pred, y_true) in enumerate(zip(y_pred_list, y_true_list)):
    axes[i, 0].imshow(np.squeeze(y_pred))
    axes[i, 0].set_title(f'Prediction {i+1}')
    axes[i, 1].imshow(np.squeeze(y_true))
    axes[i, 1].set_title(f'True {i+1}')

plt.tight_layout()
plt.show()
