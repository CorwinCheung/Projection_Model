#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

with open('dataset.pkl', 'rb') as f:
    loaded_dataset = pickle.load(f)

# Verify the shapes of the first 3D matrix and its 2D projection
print(loaded_dataset[0][0].shape)  # Shape of the 3D matrix (256, 256, 256)
print(loaded_dataset[0][1].shape)  # Shape of the 2D projection (256, 256)

class OffDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, y = self.data[idx]
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0) / 255.0
        y = torch.tensor(y, dtype=torch.float32) / 255.0
        return X, y

# Create dataset and data loaders
train_size = int(0.8 * len(loaded_dataset))
val_size = len(loaded_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(loaded_dataset, [train_size, val_size])
train_loader = DataLoader(OffDataset(train_dataset), batch_size=16, shuffle=True)
val_loader = DataLoader(OffDataset(val_dataset), batch_size=16, shuffle=False)


# In[2]:


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*256*256, 1024)
        self.fc3 = nn.Linear(1024, 256*256)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = x.view(-1, 256, 256)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = MLP().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


# In[1]:


#helpers: 
def parse_off(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    assert lines[0].strip() == 'OFF', "Not a valid OFF file"

    header = lines[1].strip().split()
    num_vertices = int(header[0])
    num_faces = int(header[1])
    
    vertices = []
    for i in range(2, 2 + num_vertices):
        vertex = list(map(float, lines[i].strip().split()))
        vertices.append(vertex)
    
    faces = []
    for i in range(2 + num_vertices, 2 + num_vertices + num_faces):
        face = list(map(int, lines[i].strip().split()[1:])) # Ignoring the first number (number of vertices in the face)
        faces.append(face)
    
    return vertices, faces

def normalize_vertices(vertices, target_dim):
    vertices = np.array(vertices)
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    
    # Translate vertices to start from (0,0,0)
    vertices -= min_coords
    
    # Scale vertices to fit within the target dimensions
    scale = target_dim / np.max(max_coords - min_coords)
    vertices *= scale
    
    return vertices

def rasterize_mesh(vertices, faces, matrix_dim):
    vertices = normalize_vertices(vertices, matrix_dim)
    
    # Ensuring the matrix dimensions
    matrix = np.zeros((matrix_dim, matrix_dim, matrix_dim), dtype=np.uint8)
    
    for face in faces:
        for vertex_idx in face:
            x, y, z = np.round(vertices[vertex_idx]).astype(int)
            x = np.clip(x, 0, matrix_dim - 1)
            y = np.clip(y, 0, matrix_dim - 1)
            z = np.clip(z, 0, matrix_dim - 1)
            matrix[x, y, z] = 1
    
    return matrix


# In[ ]:


### Run Inference on Test File
def process_off_file(file_path, matrix_dim):
    vertices, faces = parse_off(file_path)
    matrix = rasterize_mesh(vertices, faces, matrix_dim)
    return matrix

# Load the test file
test_file_path = '/mnt/data/Test Set/3D files/toilet_0443.off'  # Replace with the path to your test file
test_matrix = process_off_file(test_file_path, matrix_dim)

# Preprocess the test matrix
test_matrix_flattened = test_matrix.reshape((1, -1)) / 255.0

# Run inference
predicted_projection_flattened = model.predict(test_matrix_flattened)
predicted_projection = predicted_projection_flattened.reshape((matrix_dim, matrix_dim))

### Visualize the Predicted Projection
plt.imshow(predicted_projection, cmap='gray')
plt.title('Predicted 2D Projection')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

