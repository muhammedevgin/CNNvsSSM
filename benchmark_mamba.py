import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
import time

# --- 1. Model Definitions ---
# (Copying the exact same architecture from our previous step)
class MinimalSelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.proj_b = nn.Linear(d_model, d_state, bias=False)
        self.proj_c = nn.Linear(d_model, d_state, bias=False)
        self.proj_delta = nn.Linear(d_model, d_model, bias=False)
        self.A = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        
    def forward(self, x):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device)
        out = []
        for t in range(seq_len):
            xt = x[:, t, :]
            bt = self.proj_b(xt)
            ct = self.proj_c(xt)
            dt = F.softplus(self.proj_delta(xt))
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
        self.mamba_block = MinimalSelectiveSSM(d_model, d_state)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.mamba_block(x)
        x = self.norm(x)
        x = x.mean(dim=1)                   
        return self.head(x)

# --- 2. Profiling Utilities ---
def get_pi_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read()) / 1000.0
    except Exception as e:
        return 0.0

# --- 3. The Hardware Test ---
if __name__ == "__main__":
    # 1. Initialize the raw Python model
    model = UAVVisionMamba(d_model=64, d_state=16)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)

    # 2. Calculate Theoretical GFLOPs on the RAW model
    print("Calculating Theoretical GFLOPs...")
    flops = FlopCountAnalysis(model, dummy_input)
    gflops = flops.total() / 1e9

    print("------------------------------------------------")
    print(f"Architecture: UAV Vision Mamba (Target)")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()) / 1e6:.2f} Million")
    print(f"Complexity:   {gflops:.4f} GFLOPs")
    print("------------------------------------------------")

    # 3. NOW apply torch.compile to speed up execution
    print("Compiling model to C++ via Dynamo (This will take a minute...)")
    model = torch.compile(model)

    temp_start = get_pi_temperature()
    print(f"Initial Temperature: {temp_start:.1f} °C")
    
    # 4. Warm up ARM CPU (and trigger the actual compilation)
    print("Warming up ARM CPU and running JIT compiler...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 5. Run the Latency Benchmark
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
    print(f"Avg Inference Latency: {avg_time_ms:.2f} ms")
    print(f"Estimated FPS:         {1000 / avg_time_ms:.2f}")
    print(f"Final Temperature:     {temp_end:.1f} °C (Delta: +{temp_end - temp_start:.1f} °C)")
    print("------------------------------------------------")