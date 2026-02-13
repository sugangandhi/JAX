import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import time

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))
    def forward(self, x): return self.layers(x)

def get_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor()]) # Normalization 
    full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return DataLoader(Subset(full_train, range(50000)), batch_size=batch_size, shuffle=True), \
           DataLoader(Subset(full_train, range(50000, 60000)), batch_size=1000), \
           DataLoader(test_set, batch_size=1000)

def benchmark(batch_size):
    train_loader, val_loader, test_loader = get_data(batch_size)
    print(f"\n[PyTorch] Batch Size: {batch_size}")
    device = torch.device("cuda")
    model = SimpleMLP().to(device)
    opt = optim.SGD(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        x, y = next(iter(train_loader))
        x, y = x.view(-1, 784).to(device), y.to(device)
        start = time.time()
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        torch.cuda.synchronize() # Accurate GPU timing 
        dt = time.time() - start
        
        acc = (model(x).argmax(1) == y).float().mean().item() * 100
        val_acc = sum((model(vx.view(-1, 784).to(device)).argmax(1) == vy.to(device)).float().sum().item() 
                      for vx, vy in val_loader) / 100
        
        label = "First Epoch" if epoch == 0 else f"Steady Epoch {epoch}"
        print(f"{label}: {dt:.6f}s | Loss: {loss.item():.4f} | Accuracy: {acc:.2f}% | Val_Acc: {val_acc:.2f}%")
    
    test_acc = sum((model(tx.view(-1, 784).to(device)).argmax(1) == ty.to(device)).float().sum().item() 
                   for tx, ty in test_loader) / 100
    print(f"Final Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    for b in [64, 256, 1024]: benchmark(b)