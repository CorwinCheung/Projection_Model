import pickle

with open('dataset.pkl', 'rb') as f:
    loaded_dataset = pickle.load(f)

# Verify the shapes of the first 3D matrix and its 2D projection
print(loaded_dataset[0][0].shape)  # Shape of the 3D matrix (256, 256, 256)
print(loaded_dataset[0][1].shape)  # Shape of the 2D projection (256, 256)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx][0]
        y = self.data[idx][1]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(y, dtype=torch.float32)

# Separate the dataset into inputs (3D matrices) and outputs (2D projections)
train_data, test_data = train_test_split(loaded_dataset, test_size=0.2, random_state=42)

train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F

class CNN3Dto2D(nn.Module):
    def __init__(self):
        super(CNN3Dto2D, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128*32*32*32, 256*256)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128*32*32*32)
        x = F.relu(self.fc1(x))
        x = x.view(-1, 256, 256)
        return x

model = CNN3Dto2D()
print(model)

import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    print(f'Validation Loss: {val_loss/len(test_loader)}')

model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
print(f'Test Loss: {test_loss/len(test_loader)}')

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


def process_off_file(file_path, matrix_dim):
    vertices, faces = parse_off(file_path)
    matrix = rasterize_mesh(vertices, faces, matrix_dim)
    return matrix

# Load the test file
test_file_path = 'Test Set/3D files/toilet_0443.off'  # Replace with the path to your test file
matrix_dim = 256
test_matrix = process_off_file(test_file_path, matrix_dim)
test_matrix_exp = torch.tensor(test_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


model.eval()
with torch.no_grad():
    predicted_2d_projection = model(test_matrix_exp)
print(predicted_2d_projection.shape)  # Should be (1, 256, 256)


