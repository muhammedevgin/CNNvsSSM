#!/usr/bin/env python3
"""
CNN vs SCALED S-SSM Benchmark on Raspberry Pi 5
=================================================
Runs MobileNetV2 (passive CNN) on the ExDark Low-Light dataset,
measures GFLOPs via fvcore, latency, FPS, temperature, and compares
against a SCALED S-SSM (UAV Vision Mamba) with matched parameter count.

Original S-SSM:  d_model=64,  d_state=16, 1 block  → 0.06 M params
Scaled  S-SSM:  d_model=512, d_state=32, 6 blocks → 2.28 M params
MobileNetV2:                                        → 2.24 M params

Usage:
    source /home/mevgin/try1/uav_env/bin/activate
    python benchmark_full_scaled.py
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
import math
import selective_scan_cpp

# ============================================================
# Configuration
# ============================================================
EXDARK_DIR = "/home/mevgin/try1/exdark_dataset"
CNN_WEIGHTS_PATH = "/home/mevgin/try1/mobilenet_exdark_finetuned.pth"
IMAGE_SIZE = 224
BATCH_SIZE = 1          # Single-image inference (realistic UAV scenario)
WARMUP_ITERS = 3        # Warm-up iterations before timing
BENCHMARK_ITERS = 20    # Latency benchmark iterations (dummy input)
NUM_CLASSES = 12        # ExDark classes
DATASET_SAMPLE = 50     # Sample N images from dataset for timed run

# Scaled S-SSM parameters (matched to MobileNetV2 ~2.24M)
SCALED_D_MODEL = 512
SCALED_D_STATE = 32
SCALED_NUM_BLOCKS = 6

# Original (small) S-SSM parameters for reference
ORIG_D_MODEL = 64
ORIG_D_STATE = 16

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
# S-SSM Model Definitions
# ============================================================
class MinimalSelectiveSSM(nn.Module):
    """Single Selective State-Space block."""
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
        # Compute dynamic parameters for the entire sequence at once
        B_td = self.proj_b(x)
        C_td = self.proj_c(x)
        dt = F.softplus(self.proj_delta(x))

        # Call C++ optimized selective scan kernel
        out = selective_scan_cpp.forward(x, dt, B_td, C_td, self.A, self.D)
        
        return out


class MambaBlock(nn.Module):
    """A single Mamba layer: SSM + LayerNorm + residual connection."""
    def __init__(self, d_model, d_state):
        super().__init__()
        self.ssm = MinimalSelectiveSSM(d_model, d_state)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-norm residual connection
        return x + self.ssm(self.norm(x))


class UAVVisionMambaOriginal(nn.Module):
    """Original small S-SSM (1 block, d_model=64) — for reference."""
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


class UAVVisionMambaScaled(nn.Module):
    """
    Scaled S-SSM with parameter count matched to MobileNetV2 (~2.24M).
    Uses multiple stacked MambaBlocks with residual connections.
    
    Architecture:
        Patch Embedding → [MambaBlock × N] → LayerNorm → Global Avg Pool → Head
    """
    def __init__(self, image_size=224, patch_size=16, d_model=512,
                 d_state=32, num_blocks=6, num_classes=12):
        super().__init__()

        # Patch embedding: image → sequence of patch tokens
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size,
                                     stride=patch_size)

        # Stacked Mamba blocks with residual connections
        self.blocks = nn.Sequential(*[
            MambaBlock(d_model, d_state) for _ in range(num_blocks)
        ])

        # Final norm and classification head
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, 3, 224, 224)
        x = self.patch_embed(x)              # → (batch, d_model, 14, 14)
        x = x.flatten(2).transpose(1, 2)     # → (batch, 196, d_model)

        # Process through stacked Mamba blocks
        x = self.blocks(x)                   # → (batch, 196, d_model)

        # Global average pooling + classify
        x = self.final_norm(x)
        x = x.mean(dim=1)                    # → (batch, d_model)
        return self.head(x)


# ============================================================
# Phase 1 — CNN (MobileNetV2) Benchmark
# ============================================================
def benchmark_cnn():
    print("=" * 64)
    print("  PHASE 1: MobileNetV2 (Passive CNN) on ExDark Low-Light")
    print("=" * 64)
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

    # --- Dataset inference ---
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=EXDARK_DIR, transform=transform)
    indices = list(range(0, len(full_dataset),
                         max(1, len(full_dataset) // DATASET_SAMPLE)))[:DATASET_SAMPLE]
    dataset = Subset(full_dataset, indices)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
    print(f"  Full dataset: ExDark Low-Light ({len(full_dataset)} images, "
          f"{len(full_dataset.classes)} classes)")
    print(f"  Sampled {len(dataset)} images for timed inference")
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

    # --- Latency benchmark ---
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

    print(f"\n  --- CNN Results ---")
    print(f"  GFLOPs:                 {results['gflops']:.4f}")
    print(f"  Parameters:             {results['params_m']:.2f} M")
    print(f"  Latency (dummy, avg):   {results['avg_latency_ms']:.2f} ms")
    print(f"  FPS (dummy):            {results['fps']:.2f}")
    print(f"  Latency (dataset, avg): {results['dataset_avg_ms']:.2f} ms")
    print(f"  FPS (dataset):          {results['dataset_fps']:.2f}")
    print(f"  Accuracy (ExDark):      {results['accuracy']:.2f}%")
    print(f"  Temperature:            {results['temp_start']:.1f} → {results['temp_end']:.1f} °C "
          f"(Δ +{results['temp_delta']:.1f} °C)")
    sys.stdout.flush()

    return results


# ============================================================
# Phase 2 — Original (Small) S-SSM
# ============================================================
def benchmark_ssm_original():
    print("\n" + "=" * 64)
    print("  PHASE 2: Original S-SSM (d_model=64, 1 block) [C++ OPTIMIZED]")
    print("=" * 64)
    sys.stdout.flush()

    model = UAVVisionMambaOriginal(d_model=ORIG_D_MODEL,
                                    d_state=ORIG_D_STATE,
                                    num_classes=NUM_CLASSES)
    model.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    gflops = measure_gflops(model, dummy)
    params = count_parameters(model)
    print(f"  GFLOPs (fvcore): {gflops:.4f}")
    print(f"  Parameters:      {params:.2f} M")
    sys.stdout.flush()

    # Warm-up
    print("  Warming up …")
    sys.stdout.flush()
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(dummy)

    # Latency benchmark
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
        "gflops": gflops,
        "params_m": params,
        "avg_latency_ms": avg_latency_ms,
        "fps": 1000.0 / avg_latency_ms,
        "temp_start": temp_start,
        "temp_end": temp_end,
        "temp_delta": temp_end - temp_start,
    }

    print(f"\n  --- Original S-SSM Results ---")
    print(f"  GFLOPs:     {results['gflops']:.4f}")
    print(f"  Parameters: {results['params_m']:.2f} M")
    print(f"  Latency:    {results['avg_latency_ms']:.2f} ms")
    print(f"  FPS:        {results['fps']:.2f}")
    print(f"  Temperature: {results['temp_start']:.1f} → {results['temp_end']:.1f} °C "
          f"(Δ +{results['temp_delta']:.1f} °C)")
    sys.stdout.flush()

    return results


# ============================================================
# Phase 3 — Scaled S-SSM (Matched Parameters)
# ============================================================
def benchmark_ssm_scaled():
    print("\n" + "=" * 64)
    print(f"  PHASE 3: SCALED S-SSM (d_model={SCALED_D_MODEL}, "
          f"{SCALED_NUM_BLOCKS} blocks) [C++ OPTIMIZED]")
    print("  ↳ Parameter-matched to MobileNetV2 (~2.24M)")
    print("=" * 64)
    sys.stdout.flush()

    model = UAVVisionMambaScaled(d_model=SCALED_D_MODEL,
                                  d_state=SCALED_D_STATE,
                                  num_blocks=SCALED_NUM_BLOCKS,
                                  num_classes=NUM_CLASSES)
    model.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    gflops = measure_gflops(model, dummy)
    params = count_parameters(model)
    print(f"  GFLOPs (fvcore): {gflops:.4f}")
    print(f"  Parameters:      {params:.2f} M")
    print(f"  Architecture:    {SCALED_NUM_BLOCKS}× MambaBlock "
          f"(d={SCALED_D_MODEL}, state={SCALED_D_STATE}) + residual + pre-norm")
    sys.stdout.flush()

    # Warm-up
    print("  Warming up …")
    sys.stdout.flush()
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(dummy)

    # Latency benchmark
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
        "gflops": gflops,
        "params_m": params,
        "avg_latency_ms": avg_latency_ms,
        "fps": 1000.0 / avg_latency_ms,
        "temp_start": temp_start,
        "temp_end": temp_end,
        "temp_delta": temp_end - temp_start,
    }

    print(f"\n  --- Scaled S-SSM Results ---")
    print(f"  GFLOPs:     {results['gflops']:.4f}")
    print(f"  Parameters: {results['params_m']:.2f} M")
    print(f"  Latency:    {results['avg_latency_ms']:.2f} ms")
    print(f"  FPS:        {results['fps']:.2f}")
    print(f"  Temperature: {results['temp_start']:.1f} → {results['temp_end']:.1f} °C "
          f"(Δ +{results['temp_delta']:.1f} °C)")
    sys.stdout.flush()

    return results


# ============================================================
# Phase 4 — Three-Way Comparison Table
# ============================================================
def print_comparison_table(cnn, ssm_orig, ssm_scaled):
    cnn_g = cnn['gflops']

    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + "   COMPARISON: CNN vs C-Optimized S-SSM".center(78) + "║")
    print("╠" + "═" * 78 + "╣")

    hdr = (f"║ {'Metric':<24} │ {'CNN (MobileNetV2)':>17} │ "
           f"{'C-S-SSM (Original)':>16} │ {'C-S-SSM (Scaled)':>13} ║")
    print(hdr)
    print("╠" + "═" * 24 + "═╪═" + "═" * 17 + "═╪═"
          + "═" * 16 + "═╪═" + "═" * 13 + "═╣")

    def fmtf(v, fmt=".4f"):
        return f"{v:{fmt}}"

    rows = [
        ("GFLOPs (fvcore)",
         fmtf(cnn['gflops']),
         fmtf(ssm_orig['gflops']),
         fmtf(ssm_scaled['gflops'])),
        ("Parameters (M)",
         fmtf(cnn['params_m'], ".2f"),
         fmtf(ssm_orig['params_m'], ".2f"),
         fmtf(ssm_scaled['params_m'], ".2f")),
        ("Latency (ms)",
         fmtf(cnn['avg_latency_ms'], ".2f"),
         fmtf(ssm_orig['avg_latency_ms'], ".2f"),
         fmtf(ssm_scaled['avg_latency_ms'], ".2f")),
        ("FPS",
         fmtf(cnn['fps'], ".2f"),
         fmtf(ssm_orig['fps'], ".2f"),
         fmtf(ssm_scaled['fps'], ".2f")),
        ("Temp Start (°C)",
         fmtf(cnn['temp_start'], ".1f"),
         fmtf(ssm_orig['temp_start'], ".1f"),
         fmtf(ssm_scaled['temp_start'], ".1f")),
        ("Temp End (°C)",
         fmtf(cnn['temp_end'], ".1f"),
         fmtf(ssm_orig['temp_end'], ".1f"),
         fmtf(ssm_scaled['temp_end'], ".1f")),
        ("Temp Delta (°C)",
         f"+{cnn['temp_delta']:.1f}",
         f"+{ssm_orig['temp_delta']:.1f}",
         f"+{ssm_scaled['temp_delta']:.1f}"),
    ]

    for label, v1, v2, v3 in rows:
        print(f"║ {label:<24} │ {v1:>17} │ {v2:>16} │ {v3:>13} ║")

    # Separator for derived metrics
    print("╠" + "═" * 24 + "═╪═" + "═" * 17 + "═╪═"
          + "═" * 16 + "═╪═" + "═" * 13 + "═╣")

    # FLOP reduction vs CNN
    orig_flop_r = cnn_g / ssm_orig['gflops'] if ssm_orig['gflops'] > 0 else 0
    scaled_flop_r = cnn_g / ssm_scaled['gflops'] if ssm_scaled['gflops'] > 0 else 0
    print(f"║ {'FLOP Reduction vs CNN':<24} │ {'1.00× (base)':>17} │ "
          f"{f'{orig_flop_r:.1f}× fewer':>16} │ {f'{scaled_flop_r:.1f}× fewer':>13} ║")

    # Param ratio vs CNN
    orig_param_r = cnn['params_m'] / ssm_orig['params_m'] if ssm_orig['params_m'] > 0 else 0
    scaled_param_r = cnn['params_m'] / ssm_scaled['params_m'] if ssm_scaled['params_m'] > 0 else 0
    print(f"║ {'Param Ratio vs CNN':<24} │ {'1.00× (base)':>17} │ "
          f"{f'{orig_param_r:.1f}× fewer':>16} │ {f'{scaled_param_r:.2f}× ':>13} ║")

    # Speedup vs CNN
    orig_speed = cnn['avg_latency_ms'] / ssm_orig['avg_latency_ms'] if ssm_orig['avg_latency_ms'] > 0 else 0
    scaled_speed = cnn['avg_latency_ms'] / ssm_scaled['avg_latency_ms'] if ssm_scaled['avg_latency_ms'] > 0 else 0
    print(f"║ {'Speedup vs CNN':<24} │ {'1.00× (base)':>17} │ "
          f"{f'{orig_speed:.2f}×':>16} │ {f'{scaled_speed:.2f}×':>13} ║")

    print("╠" + "═" * 24 + "═╧═" + "═" * 17 + "═╧═"
          + "═" * 16 + "═╧═" + "═" * 13 + "═╣")

    # CNN dataset metrics
    print("║" + "  CNN Dataset Metrics (ExDark Low-Light)".ljust(78) + "║")
    print("╠" + "═" * 78 + "╣")
    print(f"║  Total Images: {cnn['dataset_total']:>6}  |  Sampled: {cnn['dataset_sampled']:>4}"
          f"  |  Avg Latency: {cnn['dataset_avg_ms']:.2f} ms  |  Accuracy: {cnn['accuracy']:.1f}%"
          .ljust(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # Footer
    print(f"\nPlatform: {platform.machine()} | "
          f"Python {platform.python_version()} | "
          f"PyTorch {torch.__version__}")
    print(f"Device:   Raspberry Pi 5 (ARM Cortex-A76, passive cooling)")
    print(f"Dataset:  ExDark Low-Light ({cnn['dataset_total']} images, {NUM_CLASSES} classes)")
    print(f"Profiler: fvcore {__import__('fvcore').__version__}")

    # Key takeaways
    print("\n" + "─" * 78)
    print("KEY TAKEAWAYS:")
    print(f"  • Scaled S-SSM matches CNN params ({ssm_scaled['params_m']:.2f}M vs "
          f"{cnn['params_m']:.2f}M) but uses {scaled_flop_r:.1f}× fewer FLOPs")
    print(f"  • CNN latency: {cnn['avg_latency_ms']:.1f}ms vs Scaled S-SSM: "
          f"{ssm_scaled['avg_latency_ms']:.1f}ms "
          f"({'CNN faster' if cnn['avg_latency_ms'] < ssm_scaled['avg_latency_ms'] else 'S-SSM faster'})")
    print(f"  • Thermal: CNN Δ+{cnn['temp_delta']:.1f}°C vs "
          f"Scaled S-SSM Δ+{ssm_scaled['temp_delta']:.1f}°C")
    if ssm_scaled['temp_delta'] < cnn['temp_delta']:
        print(f"  • S-SSM thermal advantage: "
              f"{cnn['temp_delta']/ssm_scaled['temp_delta']:.1f}× cooler"
              if ssm_scaled['temp_delta'] > 0 else
              "  • S-SSM generated negligible heat")
    print("─" * 78)

    sys.stdout.flush()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("╔" + "═" * 64 + "╗")
    print("║" + "  CNN vs SCALED S-SSM Benchmark (Parameter-Matched)".center(64) + "║")
    print("╚" + "═" * 64 + "╝")
    print(f"\nPlatform: {platform.machine()} | Python {platform.python_version()} | "
          f"PyTorch {torch.__version__}")
    print(f"Device:   CPU (no GPU)\n")
    sys.stdout.flush()

    # Run all three benchmarks
    cnn_results = benchmark_cnn()
    ssm_orig_results = benchmark_ssm_original()
    ssm_scaled_results = benchmark_ssm_scaled()

    # Print comparison
    print_comparison_table(cnn_results, ssm_orig_results, ssm_scaled_results)

    print("\n✓ Scaled benchmark complete.")
    sys.stdout.flush()
