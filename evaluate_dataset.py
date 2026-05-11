import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time

def evaluate_on_dataset(data_dir):
    # 1. Image Transformations (Handling ExDark's varying aspect ratios)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Load the Dataset from your exact path
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    print(f"Successfully loaded {len(dataset)} images from {data_dir}")
    print(f"Classes detected: {dataset.classes}")

    # 3. Load the Model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()

    # 4. Run the Evaluation Loop
    total_time = 0.0
    
    # Warmup loop to stabilize thermals before timing
    print("Warming up model...")
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    print("Starting dataset inference...")
    with torch.no_grad():
        for images, labels in dataloader:
            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()
            
            total_time += (end_time - start_time)

    # 5. Compile Hardware Metrics
    avg_time_ms = (total_time / len(dataset)) * 1000

    print("------------------------------------------------")
    print(f"Total Images Processed: {len(dataset)}")
    print(f"Average Inference Time: {avg_time_ms:.2f} ms")
    print(f"Estimated FPS: {1000 / avg_time_ms:.2f}")
    print("------------------------------------------------")
    print("Note: Classification accuracy omitted due to ImageNet vs ExDark label mismatch.")

if __name__ == "__main__":
    # Updated path to match your VS Code screenshot exactly
    evaluate_on_dataset("/home/mevgin/try1/exdark_dataset")