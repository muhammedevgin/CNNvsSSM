import torch
import torchvision.models as models
import time

def get_pi_temperature():
    """Reads the internal temperature sensor of the Raspberry Pi."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read()) / 1000.0
    except Exception as e:
        return 0.0

if __name__ == "__main__":
    # 1. Load standard MobileNet baseline
    # weights=None is fine here because we only care about measuring math speed, not accuracy
    model = models.mobilenet_v2(weights=None) 
    model.eval()

    # 2. Compile the CNN to C++ (Leveling the playing field!)
    print("Compiling CNN to C++ via Dynamo (This will take a minute...)")
    model = torch.compile(model)

    dummy_input = torch.randn(1, 3, 224, 224)

    temp_start = get_pi_temperature()
    print(f"Initial Temperature: {temp_start:.1f} °C")
    
    # 3. Warm up and trigger the C++ compilation
    print("Warming up ARM CPU and running JIT compiler...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 4. The 100-Iteration Race
    iterations = 100
    print(f"Running {iterations} inference iterations...")
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    end_time = time.perf_counter()
    
    temp_end = get_pi_temperature()
    avg_time_ms = ((end_time - start_time) / iterations) * 1000
    
    print("------------------------------------------------")
    print("COMPILED CNN BASELINE METRICS")
    print("------------------------------------------------")
    print(f"Avg Inference Latency: {avg_time_ms:.2f} ms")
    print(f"Estimated FPS:         {1000 / avg_time_ms:.2f}")
    print(f"Final Temperature:     {temp_end:.1f} °C (Delta: +{temp_end - temp_start:.1f} °C)")
    print("------------------------------------------------")