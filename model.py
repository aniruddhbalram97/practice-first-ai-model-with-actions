import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Efficient feature extraction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)  # 24x24x8
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)  # 20x20x16
        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)  # 16x16x32
        self.bn3 = nn.BatchNorm2d(32)
        
        # Global pooling to reduce parameters
        self.global_pool = nn.AdaptiveAvgPool2d(4)  # 4x4x32
        
        # Simple classifier
        self.fc = nn.Linear(32 * 4 * 4, 10)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Deep feature extraction without pooling layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(-1, 32 * 4 * 4)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,  # Slightly increased learning rate
        weight_decay=0.0001,
        betas=(0.9, 0.999)
    )
    
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        running_loss += loss.item()
        current_acc = 100. * correct / total
        
        if batch_idx % 50 == 0:
            print(f'Batch [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {running_loss/50:.4f}, '
                  f'Accuracy: {current_acc:.2f}%')
            running_loss = 0.0
    
    accuracy = 100. * correct / total
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save(model.state_dict(), f'mnist_model_{timestamp}.pth')
    
    return accuracy, model

if __name__ == "__main__":
    accuracy, _ = train_model()
    print(f"Training Accuracy: {accuracy:.2f}%") 