import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


# Define the MeshToImageNN class if not already defined
class MeshToImageNN(nn.Module):
    def __init__(self):
        super(MeshToImageNN, self).__init__()
        self.fc1 = nn.Linear(17664 * 3, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 256 * 256)
        
    def forward(self, x):
        x = x.view(-1, 17664 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, 256, 256)
        return x

# Load the trained model
model = MeshToImageNN()
state_dict = torch.load('trained_projection_model.pth')
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix if present
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

model.eval()

model.eval()

# Load and preprocess the new .off file
def preprocess_off_file(file_path):
    mesh = trimesh.load(file_path)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    
    # Pad or truncate vertices to ensure consistent tensor size
    if vertices.shape[0] > 17664:
        vertices = vertices[:17664]
    else:
        vertices = np.pad(vertices, ((0, 17664 - vertices.shape[0]), (0, 0)), 'constant')
    
    vertices = torch.tensor(vertices).float().unsqueeze(0)
    return vertices

# Inference function
def run_inference(model, file_path):
    vertices = preprocess_off_file(file_path)
    with torch.no_grad():
        predicted_image = model(vertices)
    return predicted_image

# Path to the new .off file
new_off_file = '3D files/toilet_0443.off'

# Run inference
predicted_image = run_inference(model, new_off_file)

# Visualize the predicted 2D projection
plt.figure(figsize=(6, 6))
plt.imshow(predicted_image.squeeze().numpy(), cmap='gray')
plt.title('Predicted 2D Projection')
plt.axis('off')

# Save the figure to a file
output_image_path = 'predicted_projection_toilet_0443.png'
plt.savefig(output_image_path, dpi=300)
plt.close()
print(f"Predicted 2D projection saved to {output_image_path}")
