#!/usr/bin/env python3
"""
CNN vs S-SSM Benchmark on Raspberry Pi 5
=========================================
Runs MobileNetV2 (passive CNN) on the ExDark Low-Light dataset,
measures GFLOPs via fvcore, latency, FPS, temperature, and compares
against Theoretical S-SSM (UAV Vision Mamba) targets.

Usage:
    source /home/mevgin/try1/uav_env/bin/activate
    python benchmark_full.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from fvcore.nn import FlopCountAnalysis
import time
import platform
import os
import sys

# ============================================================
# Configuration
# ============================================================
EXDARK_DIR = "/home/mevgin/try1/exdark_dataset"
CNN_WEIGHTS_PATH = "/home/mevgin/try1/mobilenet_exdark_finetuned.pth"
IMAGE_SIZE = 224
BATCH_SIZE = 1          # Single-image inference (realistic UAV scenario)
WARMUP_ITERS = 5        # Warm-up iterations before timing
BENCHMARK_ITERS = 50    # Latency benchmark iterations (dummy input)
NUM_CLASSES = 12        # ExDark classes
DATASET_SAMPLE = 100    # Sample N images from dataset for timed run

# S-SSM model hyper-parameters (must match training)
D_MODEL = 64
D_STATE = 16

# ============================================================
# Utility Functions
# ============================================================
def get_pi_temperature():
    """Read Raspberry Pi CPU temperature from thermal zone."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return float("nan")

def count_parameters(model):
    """Return total parameter count in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6

def measure_gflops(model, dummy_input):
    """Use fvcore to compute GFLOPs for a model."""
    flops = FlopCountAnalysis(model, dummy_input)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return flops.total() / 1e9

# ============================================================
# S-SSM Model Definition (Theoretical Target)
# ============================================================
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
    def __init__(self, image_size=224, patch_size=16, d_model=64,
                 d_state=16, num_classes=12):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size,
                                     stride=patch_size)
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

# ============================================================
# Phase 1 — CNN (MobileNetV2) Benchmark
# ============================================================
def benchmark_cnn():
    print("=" * 60)
    print("  PHASE 1: MobileNetV2 (Passive CNN) on ExDark Low-Light")
    print("=" * 60)
    sys.stdout.flush()

    # --- Load model ---
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    if os.path.exists(CNN_WEIGHTS_PATH):
        model.load_state_dict(torch.load(CNN_WEIGHTS_PATH,
                                         map_location="cpu",
                                         weights_only=True))
        print(f"  ✓ Loaded fine-tuned weights: {CNN_WEIGHTS_PATH}")
    else:
        print("  ⚠ Fine-tuned weights not found; using pretrained backbone")

    model.eval()

    # --- fvcore GFLOPs ---
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    cnn_gflops = measure_gflops(model, dummy)
    cnn_params = count_parameters(model)
    print(f"  GFLOPs (fvcore): {cnn_gflops:.4f}")
    print(f"  Parameters:      {cnn_params:.2f} M")
    sys.stdout.flush()

    # --- Dataset inference (sampled subset for speed) ---
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=EXDARK_DIR, transform=transform)
    # Deterministic subset for reproducibility
    indices = list(range(0, len(full_dataset),
                         max(1, len(full_dataset) // DATASET_SAMPLE)))[:DATASET_SAMPLE]
    dataset = Subset(full_dataset, indices)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
    print(f"  Full dataset: ExDark Low-Light ({len(full_dataset)} images, "
          f"{len(full_dataset.classes)} classes)")
    print(f"  Sampled {len(dataset)} images for timed inference")
    print(f"  Classes: {full_dataset.classes}")
    sys.stdout.flush()

    # Warm-up
    print("  Warming up ARM CPU …")
    sys.stdout.flush()
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(dummy)

    # --- Timed dataset inference ---
    temp_start = get_pi_temperature()
    print(f"  Temperature before dataset run: {temp_start:.1f} °C")
    print(f"  Running inference on {len(dataset)} ExDark images …")
    sys.stdout.flush()

    total_time = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            t0 = time.perf_counter()
            outputs = model(images)
            t1 = time.perf_counter()
            total_time += (t1 - t0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(dataset)}] images processed …")
                sys.stdout.flush()

    temp_after_dataset = get_pi_temperature()
    avg_dataset_ms = (total_time / len(dataset)) * 1000
    accuracy = 100.0 * correct / total

    print(f"  Dataset inference complete in {total_time:.1f}s")
    sys.stdout.flush()

    # --- Latency benchmark (dummy input, 100 iters) ---
    print(f"  Running {BENCHMARK_ITERS}-iteration latency benchmark …")
    sys.stdout.flush()
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(BENCHMARK_ITERS):
            _ = model(dummy)
    t_end = time.perf_counter()

    temp_end = get_pi_temperature()
    avg_latency_ms = ((t_end - t_start) / BENCHMARK_ITERS) * 1000

    results = {
        "gflops": cnn_gflops,
        "params_m": cnn_params,
        "avg_latency_ms": avg_latency_ms,
        "fps": 1000.0 / avg_latency_ms,
        "dataset_avg_ms": avg_dataset_ms,
        "dataset_fps": 1000.0 / avg_dataset_ms,
        "accuracy": accuracy,
        "temp_start": temp_start,
        "temp_after_dataset": temp_after_dataset,
        "temp_end": temp_end,
        "temp_delta": temp_end - temp_start,
        "dataset_total": len(full_dataset),
        "dataset_sampled": len(dataset),
    }

    print(f"\n  --- CNN Results Summary ---")
    print(f"  GFLOPs:                 {results['gflops']:.4f}")
    print(f"  Parameters:             {results['params_m']:.2f} M")
    print(f"  Latency (dummy, avg):   {results['avg_latency_ms']:.2f} ms")
    print(f"  FPS (dummy):            {results['fps']:.2f}")
    print(f"  Latency (dataset, avg): {results['dataset_avg_ms']:.2f} ms")
    print(f"  FPS (dataset):          {results['dataset_fps']:.2f}")
    print(f"  Accuracy (ExDark):      {results['accuracy']:.2f}%")
    print(f"  Temperature start:      {results['temp_start']:.1f} °C")
    print(f"  Temperature end:        {results['temp_end']:.1f} °C")
    print(f"  Temperature delta:      +{results['temp_delta']:.1f} °C")
    sys.stdout.flush()

    return results

# ============================================================
# Phase 2 — Theoretical S-SSM Targets
# ============================================================
def compute_ssm_targets():
    print("\n" + "=" * 60)
    print("  PHASE 2: Theoretical S-SSM Targets (UAV Vision Mamba)")
    print("=" * 60)
    sys.stdout.flush()

    model = UAVVisionMamba(d_model=D_MODEL, d_state=D_STATE,
                           num_classes=NUM_CLASSES)
    model.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    ssm_gflops = measure_gflops(model, dummy)
    ssm_params = count_parameters(model)

    print(f"  GFLOPs (fvcore): {ssm_gflops:.4f}")
    print(f"  Parameters:      {ssm_params:.2f} M")
    sys.stdout.flush()

    # Run latency benchmark
    print("  Warming up …")
    sys.stdout.flush()
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(dummy)

    temp_start = get_pi_temperature()
    print(f"  Running {BENCHMARK_ITERS}-iteration latency benchmark …")
    sys.stdout.flush()
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(BENCHMARK_ITERS):
            _ = model(dummy)
    t_end = time.perf_counter()
    temp_end = get_pi_temperature()

    avg_latency_ms = ((t_end - t_start) / BENCHMARK_ITERS) * 1000

    results = {
        "gflops": ssm_gflops,
        "params_m": ssm_params,
        "avg_latency_ms": avg_latency_ms,
        "fps": 1000.0 / avg_latency_ms,
        "temp_start": temp_start,
        "temp_end": temp_end,
        "temp_delta": temp_end - temp_start,
    }

    print(f"\n  --- S-SSM Theoretical Target Summary ---")
    print(f"  GFLOPs:            {results['gflops']:.4f}")
    print(f"  Parameters:        {results['params_m']:.2f} M")
    print(f"  Latency (avg):     {results['avg_latency_ms']:.2f} ms")
    print(f"  FPS:               {results['fps']:.2f}")
    print(f"  Temperature start: {results['temp_start']:.1f} °C")
    print(f"  Temperature end:   {results['temp_end']:.1f} °C")
    print(f"  Temperature delta: +{results['temp_delta']:.1f} °C")
    sys.stdout.flush()

    return results

# ============================================================
# Phase 3 — Comparison Table
# ============================================================
def print_comparison_table(cnn, ssm):
    flop_ratio = cnn['gflops'] / ssm['gflops'] if ssm['gflops'] > 0 else 0
    speed_ratio = cnn['avg_latency_ms'] / ssm['avg_latency_ms'] if ssm['avg_latency_ms'] > 0 else 0
    param_ratio = cnn['params_m'] / ssm['params_m'] if ssm['params_m'] > 0 else 0

    print("\n")
    print("╔" + "═" * 62 + "╗")
    print("║" + "  COMPARISON: Passive CNN on Pi  vs  Theoretical S-SSM".center(62) + "║")
    print("╠" + "═" * 62 + "╣")

    hdr = f"║ {'Metric':<28} │ {'CNN (MobileNetV2)':>14} │ {'S-SSM (Mamba)':>13} ║"
    print(hdr)
    print("╠" + "═" * 28 + "═╪═" + "═" * 14 + "═╪═" + "═" * 13 + "═╣")

    rows = [
        ("GFLOPs (fvcore)",      f"{cnn['gflops']:.4f}",         f"{ssm['gflops']:.4f}"),
        ("Parameters (M)",       f"{cnn['params_m']:.2f}",       f"{ssm['params_m']:.2f}"),
        ("Latency (ms)",         f"{cnn['avg_latency_ms']:.2f}", f"{ssm['avg_latency_ms']:.2f}"),
        ("FPS",                  f"{cnn['fps']:.2f}",            f"{ssm['fps']:.2f}"),
        ("Temp Start (°C)",      f"{cnn['temp_start']:.1f}",     f"{ssm['temp_start']:.1f}"),
        ("Temp End (°C)",        f"{cnn['temp_end']:.1f}",       f"{ssm['temp_end']:.1f}"),
        ("Temp Delta (°C)",      f"+{cnn['temp_delta']:.1f}",    f"+{ssm['temp_delta']:.1f}"),
        ("FLOP Reduction",       "1.00× (base)",                 f"{flop_ratio:.2f}× fewer"),
        ("Param Reduction",      "1.00× (base)",                 f"{param_ratio:.1f}× fewer"),
        ("Speedup",              "1.00× (base)",                 f"{speed_ratio:.2f}×"),
    ]

    for label, v1, v2 in rows:
        print(f"║ {label:<28} │ {v1:>14} │ {v2:>13} ║")

    print("╠" + "═" * 28 + "═╧═" + "═" * 14 + "═╧═" + "═" * 13 + "═╣")

    # CNN-only dataset metrics
    print("║" + "  CNN Dataset Metrics (ExDark Low-Light)".ljust(62) + "║")
    print("╠" + "═" * 62 + "╣")
    print(f"║ {'Total Dataset Images':<28}   {cnn['dataset_total']:>14}              ║")
    print(f"║ {'Sampled for Benchmark':<28}   {cnn['dataset_sampled']:>14}              ║")
    print(f"║ {'Dataset Avg Latency (ms)':<28}   {cnn['dataset_avg_ms']:>14.2f}              ║")
    print(f"║ {'Dataset FPS':<28}   {cnn['dataset_fps']:>14.2f}              ║")
    print(f"║ {'ExDark Accuracy (%)':<28}   {cnn['accuracy']:>14.2f}              ║")
    print("╚" + "═" * 62 + "╝")

    # Hardware info
    print(f"\nPlatform: {platform.machine()} | "
          f"Python {platform.python_version()} | "
          f"PyTorch {torch.__version__}")
    print(f"Device:   Raspberry Pi 5 (ARM Cortex-A76, passive cooling)")
    print(f"Dataset:  ExDark Low-Light ({cnn['dataset_total']} total images, "
          f"{NUM_CLASSES} classes)")
    print(f"Profiler: fvcore {__import__('fvcore').__version__}")
    sys.stdout.flush()

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"Platform: {platform.machine()} | Python {platform.python_version()} | "
          f"PyTorch {torch.__version__}")
    print(f"Device:   CPU (no GPU)\n")
    sys.stdout.flush()

    cnn_results = benchmark_cnn()
    ssm_results = compute_ssm_targets()
    print_comparison_table(cnn_results, ssm_results)

    print("\n✓ Benchmark complete.")
    sys.stdout.flush()
