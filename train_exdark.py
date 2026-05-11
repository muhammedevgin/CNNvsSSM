import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import time

def fine_tune_mobilenet(data_dir, num_epochs=3):
    # 1. Transforms (Added RandomHorizontalFlip to prevent overfitting)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Load Dataset and Create Train/Val Splits (80% Train, 20% Validation)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    num_classes = len(full_dataset.classes)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # batch_size=16 is usually the sweet spot for Pi 5 memory
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    print(f"Dataset Split: {train_size} Training | {val_size} Validation")
    print(f"Classes ({num_classes}): {full_dataset.classes}")

    # 3. Load Model & Freeze the Backbone
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False # Freeze all layers

    # 4. Replace the Classification Head (Requires grad by default)
    # MobileNetV2's classifier is a Sequential block; we replace the final Linear layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # 5. Define Loss and Optimizer (Only optimizing the new classifier head)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

    # 6. The Training Loop
    print("\nStarting Training (Linear Probing)...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i+1) % 50 == 0:
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

    # 7. Save the fine-tuned weights
    torch.save(model.state_dict(), "mobilenet_exdark_finetuned.pth")
    print("Training complete. Model saved as 'mobilenet_exdark_finetuned.pth'")

if __name__ == "__main__":
    # Point this to your ExDark directory
    fine_tune_mobilenet("/home/mevgin/try1/exdark_dataset", num_epochs=5)