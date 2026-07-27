# Darktable Mojo Benchmarks

This directory contains performance benchmarks and validation tools for darktable module kernels ported to Mojo.

## Subprojects

- **[Sigmoid](./sigmoid)**: Benchmarks and parity checks for the sigmoid module.
- **[Blurs](./blurs)**: Benchmarks for the blurs convolution module.

## Prerequisites

- **Mojo**: Required for running `.mojo` benchmarks.
- **OpenCL**: Required for hardware acceleration on GPU (headers and libraries).
- **Pixi**: Used for environment and dependency management.

## Environment Setup

Initialize the environment using Pixi in the root `benchmark/` directory:

```bash
pixi install
```
