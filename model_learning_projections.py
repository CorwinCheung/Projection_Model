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

class MeshDataset(Dataset):
    def __init__(self, mesh_dir, img_dir, transform=None):
        self.mesh_dir = mesh_dir
        self.img_dir = img_dir
        self.transform = transform
        self.mesh_files = [f for f in os.listdir(mesh_dir)]
        self.img_files = [f for f in os.listdir(img_dir)]

    def __len__(self):
        return len(self.mesh_files)
    
    def __getitem__(self, idx):
        mesh_file = self.mesh_files[idx]
        img_file = mesh_file.replace('.off','.png')

        mesh_path = os.path.join(self.mesh_dir, mesh_file)
        img_path = os.path.join(self.img_dir, img_file)
        
        # Load mesh
        mesh = trimesh.load(mesh_path)
        vertices = np.array(mesh.vertices, dtype=np.float32)

        # Pad or truncate vertices to ensure consistent tensor size
        if vertices.shape[0] > 17664:
            vertices = vertices[:17664]
        else:
            vertices = np.pad(vertices, ((0, 17664 - vertices.shape[0]), (0, 0)), 'constant')
        
        vertices = torch.tensor(vertices)
        
        # Load image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        
        return vertices, image
    
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

model = MeshToImageNN()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    for i, (vertices, images) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(vertices)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

print('Training complete.')
