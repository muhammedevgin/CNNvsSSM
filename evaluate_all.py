import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from benchmark_full_Coptimized import UAVVisionMambaOriginal

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(root="/home/mevgin/try1/exdark_dataset", transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
_, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

print("Evaluating MobileNetV2...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 12)
model.load_state_dict(torch.load("/home/mevgin/try1/mobilenet_exdark_finetuned.pth", map_location="cpu", weights_only=True))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"MobileNetV2 Validation Accuracy: {100.0 * correct / total:.2f}%")

print("Evaluating Original S-SSM...")
ssm = UAVVisionMambaOriginal(d_model=64, d_state=16, num_classes=12)
try:
    ssm.load_state_dict(torch.load("/home/mevgin/try1/uav_mamba_exdark.pth", map_location="cpu", weights_only=True))
    ssm.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = ssm(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Original S-SSM Validation Accuracy: {100.0 * correct / total:.2f}%")
except Exception as e:
    print(f"Failed to load/eval S-SSM: {e}")

