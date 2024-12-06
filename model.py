import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)  # 28x28x4
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)  # 28x28x8
        self.bn2 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2, 2)  # 14x14x8
        
        self.conv3 = nn.Conv2d(8, 12, kernel_size=3, padding=1)  # 14x14x12
        self.bn3 = nn.BatchNorm2d(12)
        self.conv4 = nn.Conv2d(12, 16, kernel_size=3, padding=1)  # 14x14x16
        self.bn4 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)  # 7x7x16
        
        self.fc1 = nn.Linear(16 * 7 * 7, 24)
        self.fc2 = nn.Linear(24, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        x = x.view(-1, 16 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
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
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    total_steps = len(train_loader)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,
        steps_per_epoch=total_steps,
        epochs=1,
        pct_start=0.2,
        div_factor=10,
        final_div_factor=100,
        anneal_strategy='cos'
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        running_loss += loss.item()
        current_acc = 100. * correct / total
        
        if batch_idx % 50 == 0:
            print(f'Batch [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {running_loss/50:.4f}, '
                  f'Accuracy: {current_acc:.2f}%, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
            running_loss = 0.0
    
    accuracy = 100. * correct / total
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save(model.state_dict(), f'mnist_model_{timestamp}.pth')
    
    return accuracy, model

if __name__ == "__main__":
    accuracy, _ = train_model()
    print(f"Training Accuracy: {accuracy:.2f}%") 