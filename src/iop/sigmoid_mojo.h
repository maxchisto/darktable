#pragma once
#include <stdint.h>

// Opaque context (hides Mojo DeviceContext internals)
typedef void* SigmoidMojoCtx;

// Parameters passed per-frame (mirrors SigmoidMojoParams in lib.mojo)
typedef struct {
    float white_target;
    float black_target;
    float paper_exposure;
    float film_fog;
    float film_power;
    float paper_power;
    float hue_preservation;
    // 4x4 float matrices (row-major, 16 floats each)
    float pipe_to_base[16];
    float base_to_rendering[16];
    float rendering_to_pipe[16];
} SigmoidMojoParams;

// Lifecycle
SigmoidMojoCtx sigmoid_mojo_init(int use_gpu);  // 1=GPU, 0=CPU
void            sigmoid_mojo_destroy(SigmoidMojoCtx ctx);

// Processing  (in/out are RGBA float32, stride = width * 4 * sizeof(float))
void sigmoid_mojo_rgb_ratio(
    SigmoidMojoCtx ctx,
    const float* in, float* out,
    int width, int height,
    const SigmoidMojoParams* p);

void sigmoid_mojo_per_channel(
    SigmoidMojoCtx ctx,
    const float* in, float* out,
    int width, int height,
    const SigmoidMojoParams* p);
