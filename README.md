# Edge AI Benchmarking: CNN vs. SSM on Raspberry Pi 5

## Overview
This repository provides a hardware-specific comparative analysis of Convolutional Neural Networks (CNNs) and modern State Space Models (SSMs). 
The core objective is to evaluate the architectural trade-offs—such as inference speed, parameter efficiency, and memory footprint—when deploying these models in resource-constrained edge computing environments.

## Key Features
* **Edge Deployment:** Custom implementations of baseline CNN and SSM architectures evaluated directly on ARM-based hardware.
* **Hardware Profiling:** Detailed benchmarking focused on computational overhead and active memory utilization within an 8GB RAM constraint.
* **Architectural Trade-offs:** Analysis of how traditional convolutions compare to sequence modeling (SSMs) in a non-GPU, edge-computing scenario.

## Tech Stack
* **Hardware Target:** Raspberry Pi 5 (8GB RAM) / ARM Cortex-A76
* **Language:** Python
* **Framework:** PyTorch
* **Domain:** Edge AI / Embedded Machine Learning

## Installation & Usage
Clone the repository to your Raspberry Pi environment:
```bash
git clone [https://github.com/muhammedevgin/CNNvsSSM.git](https://github.com/muhammedevgin/CNNvsSSM.git)
cd CNNvsSSM
