#!/usr/bin/env python
# coding: utf-8

# In[19]:


import trimesh
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import json
import torch_geometric


class MeshDataset(Dataset):
    def __init__(self, mesh_dir, proj_dir, max_vertices=17664, max_points=20000, transform=None):
        self.mesh_dir = mesh_dir
        self.proj_dir = proj_dir
        self.max_vertices = max_vertices
        self.max_points = max_points
        self.transform = transform
        self.mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.off')]
        self.proj_files = [f for f in os.listdir(proj_dir) if f.endswith('.json')]

    def __len__(self):
        return len(self.mesh_files)
    
    def __getitem__(self, idx):
        mesh_file = self.mesh_files[idx]
        proj_file = mesh_file.replace('.off','.json')

        mesh_path = os.path.join(self.mesh_dir, mesh_file)
        proj_path = os.path.join(self.proj_dir, proj_file)
        
        # Load mesh
        mesh = trimesh.load(mesh_path)
        vertices = np.array(mesh.vertices, dtype=np.float32)

        
        # Normalize vertices to range [-1, 1]
        vertices_min = vertices.min(axis=0)
        vertices_max = vertices.max(axis=0)
        vertices = 2 * (vertices - vertices_min) / (vertices_max - vertices_min) - 1

        # Convert vertices to tensor
        vertices = torch.tensor(vertices, dtype=torch.float32)

        # Pad or truncate vertices
        if vertices.shape[0] > self.max_vertices:
            vertices = vertices[:self.max_vertices]
        else:
            vertices = F.pad(vertices, (0, 0, 0, self.max_vertices - vertices.shape[0]))
        
        # Load x and y coordinates from JSON file
        with open(proj_path, 'r') as f:
            projection_data = json.load(f)
        x = np.array(projection_data['x'])
        y = np.array(projection_data['y'])
        
        # Normalize xy to range [-1, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x = 2 * (x - x_min) / (x_max - x_min) - 1
        y = 2 * (y - y_min) / (y_max - y_min) - 1 

        # Convert x and y to a single tensor
        xy = torch.tensor(np.stack((x, y), axis=1), dtype=torch.float32) 
        if xy.shape[0] > self.max_points:
            xy = xy[:self.max_points]
        else:
            xy = F.pad(xy, (0, 0, 0, self.max_points - xy.shape[0])) 
        
        return vertices, xy


# In[20]:


mesh_dir = '3D files'
img_dir = '2D projections'

# Transform for images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Dataset and DataLoader
dataset = MeshDataset(mesh_dir, img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(dataset)
print(dataloader)

# In[23]:


import matplotlib.pyplot as plt
from pythreejs import *
from IPython.display import display


data_iter = iter(dataloader)
vertices, xy = next(data_iter)

# Select the first element
vertices_example = vertices[0]
xy_example = xy[0]

# Print type, shape, and average value for the 3D shape (vertices) and the 2D projection (coordinates)
print("Vertices Type: ", type(vertices_example))
print("Vertices Shape: ", vertices_example.shape)
print("Vertices Average Value: ", vertices_example.mean().item())

print("XY Type: ", type(xy_example))
print("XY Shape: ", xy_example.shape)
print("XY Average Value: ", xy_example.mean().item())

# Extract x and y coordinates from xy tensor
x = xy_example[:, 0].numpy()
y = xy_example[:, 1].numpy()

# Plot the 2D scatter plot of x and y coordinates
plt.figure(figsize=(10, 10))
plt.scatter(x, y, s=1)
plt.title('2D Projection of Image Vertices')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
#plt.show()


# In[25]:


vertices_np = vertices_example.numpy()
faces = np.arange(vertices_np.shape[0], dtype=np.uint32)

geometry = BufferGeometry(
    attributes={
        'position': BufferAttribute(vertices_np, normalized=False),
        'index': BufferAttribute(faces, normalized=False)
    }
)

material = MeshBasicMaterial(color='red', wireframe=True)
three_mesh = Mesh(geometry, material)

# Set up the scene and renderer
camera = PerspectiveCamera(position=[5, 5, 5], up=[0, 0, 1], aspect=1, fov=60)
camera.lookAt([0, 0, 0])

scene = Scene(children=[three_mesh, AmbientLight(color='#cccccc')])
renderer = Renderer(camera=camera, scene=scene, controls=[OrbitControls(controlling=camera)],
                    width=800, height=600)

# Display the rendered object
display(renderer)


# In[27]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class MeshGNN(torch.nn.Module):
    def __init__(self, max_vertices=17664, max_points=20000):
        super(MeshGNN, self).__init__()
        self.max_vertices = max_vertices
        self.max_points = max_points
        
        self.conv1 = nn.Conv1d(3, 32, kernel_size=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1)
 
        self.fc1 = nn.Linear(128*max_vertices, 512)
        self.fc2 = nn.Linear(512, max_points * 2)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, 3, max_vertices)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
 
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, self.max_points, 2)
        return x

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MeshGNN().to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print("Using DataParallel")

import chamferdist

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.chamfer_dist = chamferdist.ChamferDistance()

    def forward(self, pred, target):
        dist1, dist2 = self.chamfer_dist(pred, target)
        loss = (torch.mean(dist1) + torch.mean(dist2)) / 2.0
        return loss

# Loss and optimizer
criterion = ChamferLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f'Starting epoch {epoch + 1}')
    epoch_loss=0
    for i, (vertices, xy) in enumerate(dataloader):
        vertices = vertices.to(device)
        xy = xy.to(device)
        optimizer.zero_grad()
        outputs = model(vertices)
        loss = criterion(outputs, xy)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

        if (i + 1) % 4 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    avg_loss = epoch_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

print('Training complete.')

# Save the model
#torch.save(model.state_dict(), 'mesh_to_image_model.pth')


# In[32]:


# Load the trained model for inference
model = MeshGNN().to(device)
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

test_mesh_path = 'Test Set/3D files/toilet_0443.off'
test_mesh = trimesh.load(test_mesh_path)
vertices = np.array(test_mesh.vertices, dtype=np.float32)

# Normalize vertices to range [-1, 1]
vertices_min = vertices.min(axis=0)
vertices_max = vertices.max(axis=0)
vertices = 2 * (vertices - vertices_min) / (vertices_max - vertices_min) - 1

# Convert vertices to tensor
vertices = torch.tensor(vertices, dtype=torch.float32)

# Pad or truncate vertices to ensure consistent tensor size
max_vertices = 17664
if vertices.shape[0] > max_vertices:
    vertices = vertices[:max_vertices]
else:
    vertices = np.pad(vertices, ((0, max_vertices - vertices.shape[0]), (0, 0)), 'constant')

# Ensure the tensor has the correct shape (batch_size, max_vertices, 3)
vertices = torch.tensor(vertices).unsqueeze(0).to(device)

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
plt.savefig('v2_predicted_2D_projection.png')  # Save the plot
plt.show()


# In[ ]:




