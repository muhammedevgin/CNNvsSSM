import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import time

# --- 1. S-SSM Architecture ---
class MinimalSelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.proj_b = nn.Linear(d_model, d_state, bias=False)
        self.proj_c = nn.Linear(d_model, d_state, bias=False)
        self.proj_delta = nn.Linear(d_model, d_model, bias=False)
        
        # STABILIZATION 1: Negative Initialization for A
        # This keeps the exponential decay stable
        A_init = -torch.rand(d_model, d_state) - 0.1 
        self.A = nn.Parameter(A_init)
        
        self.D = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device)
        out = []
        for t in range(seq_len):
            xt = x[:, t, :]
            bt = self.proj_b(xt)
            ct = self.proj_c(xt)
            
            # STABILIZATION 2: Clamp the softplus output
            # Prevents dt from becoming massive before the torch.exp
            dt = F.softplus(self.proj_delta(xt))
            dt = torch.clamp(dt, min=0.001, max=0.5) 
            
            dA = torch.exp(torch.einsum('bd,ds->bds', dt, self.A))
            dB = torch.einsum('bd,bs->bds', dt, bt)
            h = dA * h + torch.einsum('bds,bd->bds', dB, xt)
            yt = torch.einsum('bds,bs->bd', h, ct) + self.D * xt
            out.append(yt)
        return torch.stack(out, dim=1)

class UAVVisionMamba(nn.Module):
    def __init__(self, image_size=224, patch_size=16, d_model=64, d_state=16, num_classes=12):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.mamba_block = MinimalSelectiveSSM(d_model, d_state=d_state)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.mamba_block(x)
        x = self.norm(x)
        x = x.mean(dim=1)                   
        return self.head(x)

# --- 2. The Training Loop ---
def train_sssm(data_dir, num_epochs=3):
    # Same augmentation as the CNN baseline to keep it fair
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    num_classes = len(full_dataset.classes)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Batch size kept low to respect Pi 5 RAM limits during backpropagation
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Initialize model from scratch
    model = UAVVisionMamba(d_model=64, d_state=16, num_classes=num_classes)
    
    # We must train ALL parameters (Unlike the CNN where we froze the backbone)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    print("\nStarting S-SSM Training (From Scratch)...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # STABILIZATION 3: Gradient Clipping
            # Forces massive gradients back down to a safe maximum size (1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i+1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_time = time.time() - start_time
        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Summary: Time: {epoch_time:.1f}s | Val Accuracy: {val_acc:.2f}%\n")

    torch.save(model.state_dict(), "uav_mamba_exdark.pth")
    print("Training complete. Model saved as 'uav_mamba_exdark.pth'")

if __name__ == "__main__":
    train_sssm("/home/mevgin/try1/exdark_dataset", num_epochs=3)