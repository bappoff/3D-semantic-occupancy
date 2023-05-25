import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the .pkl file
with open('/work/scitas-share/voxformer/VoxFormer/3D-semantic-occupancy/results/test.pkl', 'rb') as file:
    data = pickle.load(file)

# Extract the arrays
y_pred = data['y_pred']
y_true = data['y_true']

# Perform visualization using the extracted arrays
# (Example visualization using matplotlib)
fig, axes = plt.subplots(nrows=2, ncols=1)

axes[0].imshow(np.squeeze(y_pred))
axes[0].set_title('Predicted')
axes[0].axis('off')

axes[1].imshow(np.squeeze(y_true))
axes[1].set_title('Ground Truth')
axes[1].axis('off')

plt.tight_layout()
plt.show()
