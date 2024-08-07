import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class MeshToImageNN(nn.Module):
    def __init__(self, max_vertices=17664, max_points=20000):
        super(MeshToImageNN, self).__init__()
        self.max_vertices = max_vertices
        self.max_points = max_points
        self.fc1 = nn.Linear(max_vertices * 3, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, max_points * 2)
        
    def forward(self, x):
        x = x.view(-1, self.max_vertices * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.max_points, 2)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MeshToImageNN().to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Load the trained model for inference
model = MeshToImageNN().to(device)
state_dict = torch.load('mesh_to_image_model.pth')
if 'module.' in list(state_dict.keys())[0]:
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(state_dict)
model.eval()

# Load the test mesh
test_mesh_path = 'Test Set/3D files/toilet_0443.off'
test_mesh = trimesh.load(test_mesh_path)
vertices = np.array(test_mesh.vertices, dtype=np.float32)

# Pad or truncate vertices to ensure consistent tensor size
max_vertices = 17664
if vertices.shape[0] > max_vertices:
    vertices = vertices[:max_vertices]
else:
    vertices = np.pad(vertices, ((0, max_vertices - vertices.shape[0]), (0, 0)), 'constant')

vertices = torch.tensor(vertices).unsqueeze(0).to(device)  # Add batch dimension

# Run inference
with torch.no_grad():
    predicted_xy = model(vertices).squeeze(0).cpu().numpy()

# Extract x and y coordinates from predicted_xy tensor
x_pred = predicted_xy[:, 0]
y_pred = predicted_xy[:, 1]

# Plot the 2D scatter plot of predicted x and y coordinates
plt.figure(figsize=(10, 10))
plt.scatter(x_pred, y_pred, s=1)
plt.title('Predicted 2D Projection of Mesh Vertices')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.savefig('predicted_projection_toilet_0443.png')  # Save the plot
plt.show()

