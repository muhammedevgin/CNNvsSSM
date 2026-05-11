import torch
import torchvision.models as models
from fvcore.nn import FlopCountAnalysis
import time

def get_pi_temperature():
    """Reads the internal temperature sensor of the Raspberry Pi."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_c = int(f.read()) / 1000.0
        return temp_c
    except Exception as e:
        return f"Error reading temp: {e}"

def profile_model():
    # 1. Load standard MobileNet baseline
    model = models.mobilenet_v2(pretrained=False)
    model.eval()

    # 2. Create a dummy input mimicking your Low Light dataset resolution (e.g., 224x224 RGB)
    # Adjust the shape (batch_size, channels, height, width) as needed for your specific UAV data
    dummy_input = torch.randn(1, 3, 224, 224)

    # 3. Measure GFLOPs with fvcore
    flops = FlopCountAnalysis(model, dummy_input)
    gflops = flops.total() / 1e9
    print(f"Theoretical Complexity: {gflops:.4f} GFLOPs")

    # 4. Measure Temperature & Inference Time
    print(f"Initial Temperature: {get_pi_temperature():.1f} °C")
    
    # Warm up the processor
    for _ in range(10):
        _ = model(dummy_input)

    start_time = time.time()
    
    # Run a mock inference loop (replace with your actual validation loop over the Low Light dataset)
    iterations = 100
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            
    end_time = time.time()
    
    avg_inference_time = ((end_time - start_time) / iterations) * 1000 # in milliseconds
    print(f"Average Inference Time: {avg_inference_time:.2f} ms")
    print(f"Post-Inference Temperature: {get_pi_temperature():.1f} °C")

if __name__ == "__main__":
    profile_model()