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

        # Pad or truncate vertices to ensure consistent tensor size
        if vertices.shape[0] > self.max_vertices:
            vertices = vertices[:self.max_vertices]
        else:
            vertices = np.pad(vertices, ((0, self.max_vertices - vertices.shape[0]), (0, 0)), 'constant')
        
        vertices = torch.tensor(vertices)
        
        # Load x and y coordinates from JSON file
        with open(proj_path, 'r') as f:
            projection_data = json.load(f)
        x = np.array(projection_data['x'])
        y = np.array(projection_data['y'])
        
        # Convert x and y to a single tensor and pad/truncate to ensure consistent size
        xy = np.stack((x, y), axis=1)
        if xy.shape[0] > self.max_points:
            xy = xy[:self.max_points]
        else:
            xy = np.pad(xy, ((0, self.max_points - xy.shape[0]), (0, 0)), 'constant')
        
        xy = torch.tensor(xy, dtype=torch.float32)

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
plt.show()


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


# In[28]:


# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dataset and DataLoader setup
mesh_dir = '3D files'
proj_dir = '2D projections'

# Transform for images (if needed)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Dataset and DataLoader
dataset = MeshDataset(mesh_dir, proj_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    for i, (vertices, xy) in enumerate(dataloader):
        vertices = vertices.to(device)
        xy = xy.to(device)
        optimizer.zero_grad()
        outputs = model(vertices)
        loss = criterion(outputs, xy)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

print('Training complete.')

# Save the model
torch.save(model.state_dict(), 'mesh_to_image_model.pth')


# In[32]:


# Load the trained model for inference
device = "cpu"
model = MeshToImageNN().to(device)
model.load_state_dict(torch.load('mesh_to_image_model.pth'))
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

vertices = torch.tensor(vertices).unsqueeze(0)  # Add batch dimension

# Run inference
with torch.no_grad():
    predicted_xy = model(vertices).squeeze(0).numpy()

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
plt.savefig('predicted_2D_projection.png')  # Save the plot
plt.show()


# In[ ]:




