import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Transform
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

# Dataset
dataset = datasets.ImageFolder('dataset', transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# CNN Model
class SignModel(nn.Module):
    def __init__(self, num_classes):
        super(SignModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*14*14, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SignModel(len(dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):
    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

torch.save(model.state_dict(), 'model/sign_model.pth')
print("Model saved!")