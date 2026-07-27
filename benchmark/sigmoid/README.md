# Sigmoid Kernel Benchmarks

This directory contains benchmarks and parity checks for the darktable `sigmoid` module kernels, comparing the traditional **OpenCL (C)** implementation against the **Mojo GPU** implementation.

## Prerequisites

- **Mojo**: Required for running `.mojo` benchmarks.
- **OpenCL**: Required for hardware acceleration on GPU (headers and libraries).
- **Pixi**: Used for environment and dependency management.

## Environment Setup

Initialize the environment using Pixi:

```bash
pixi install
# or enter the shell
pixi shell
```

## Running Benchmarks

### 1. OpenCL Baseline (C)
The C benchmark measures the performance of the original OpenCL kernels.

```bash
make run
```

### 2. Mojo GPU Benchmark
The Mojo benchmark measures the performance of the ported kernels using Mojo's GPU abstraction.

```bash
pixi run mojo sigmoid_benchmark_gpu.mojo
```

## Parity Validation

To ensure the Mojo implementation produces numerically identical results to the OpenCL baseline, run the parity check script:

```bash
pixi run python validate_parity.py
```

This script:
1. Compiles and runs the OpenCL parity check (`parity_check.c`).
2. Runs the Mojo GPU benchmark.
3. Compares sampled pixels across both implementations (RGB Ratio and Per-Channel modes).
4. Reports "PASS" if the results match within a tolerance of $10^{-6}$.

## Files

- `benchmark_sigmoid.c`: Main C-based OpenCL benchmark.
- `sigmoid_benchmark_gpu.mojo`: Mojo implementation and benchmark.
- `parity_check.c`: Minimal OpenCL runner for numerical validation.
- `validate_parity.py`: Automated comparison tool.
- `Makefile`: Build instructions for C/OpenCL binaries.
- `pixi.toml`: Project dependencies (Mojo, Python, etc.).
