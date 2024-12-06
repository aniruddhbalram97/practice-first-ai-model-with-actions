import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)  # 26x26x8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)  # 24x24x16
        self.pool = nn.MaxPool2d(2, 2)  # 12x12x16
        self.fc1 = nn.Linear(16 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 12 * 12)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    
    # Save model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save(model.state_dict(), f'mnist_model_{timestamp}.pth')
    
    return accuracy, model

if __name__ == "__main__":
    accuracy, _ = train_model()
    print(f"Training Accuracy: {accuracy:.2f}%") 