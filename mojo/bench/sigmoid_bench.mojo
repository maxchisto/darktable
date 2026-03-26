from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from std.utils import Index, IndexList
from std.math import sqrt, pow, max, min
from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.algorithm.functional import elementwise
from iop.sigmoid.kernels import apply_sigmoid_rgb_ratio, apply_sigmoid_per_channel

# Configuration
comptime WIDTH = 6016
comptime HEIGHT = 4016
comptime CHANNELS = 4
comptime IMAGE_LAYOUT = Layout.row_major(HEIGHT, WIDTH, CHANNELS)
comptime DTYPE = DType.float32

fn run_sigmoid_rgb_ratio(
    ctx: DeviceContext,
    output: LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin],
    input: LayoutTensor[DTYPE, IMAGE_LAYOUT, ImmutAnyOrigin],
    white_target: Float32,
    black_target: Float32,
    paper_exp: Float32,
    film_fog: Float32,
    film_power: Float32,
    paper_power: Float32,
    num_pixels: Int,
) raises:
    @parameter
    @always_inline
    fn rgb_ratio_closure[
        width: Int, rank: Int, alignment: Int
    ](indices: IndexList[rank]) capturing -> None:
        var px_idx = indices[0]
        var y = px_idx // WIDTH
        var x = px_idx % WIDTH
        var pix = input.load[width=4](Index(y, x, 0))
        var res = apply_sigmoid_rgb_ratio(
            pix[0], pix[1], pix[2], pix[3],
            white_target, black_target, paper_exp, film_fog, film_power, paper_power
        )
        output.store[width=4](Index(y, x, 0), res)

    elementwise[rgb_ratio_closure, 1, target="gpu"](num_pixels, ctx)


fn run_sigmoid_per_channel(
    ctx: DeviceContext,
    output: LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin],
    input: LayoutTensor[DTYPE, IMAGE_LAYOUT, ImmutAnyOrigin],
    white_target: Float32,
    paper_exp: Float32,
    film_fog: Float32,
    contrast_power: Float32,
    skew_power: Float32,
    hue_preservation: Float32,
    pipe_to_base: SIMD[DType.float32, 16],
    base_to_rendering: SIMD[DType.float32, 16],
    rendering_to_pipe: SIMD[DType.float32, 16],
    num_pixels: Int,
) raises:
    @parameter
    @always_inline
    fn per_channel_closure[
        width: Int, rank: Int, alignment: Int
    ](indices: IndexList[rank]) capturing -> None:
        var px_idx = indices[0]
        var y = px_idx // WIDTH
        var x = px_idx % WIDTH
        var pix = input.load[width=4](Index(y, x, 0))
        var res = apply_sigmoid_per_channel(
            pix[0], pix[1], pix[2], pix[3],
            white_target, paper_exp, film_fog, contrast_power, skew_power, hue_preservation,
            pipe_to_base, base_to_rendering, rendering_to_pipe
        )
        output.store[width=4](Index(y, x, 0), res)

    elementwise[per_channel_closure, 1, target="gpu"](num_pixels, ctx)


fn main() raises:
    print("Mojo Sigmoid Benchmark - New Structure")
    var total_floats = HEIGHT * WIDTH * CHANNELS
    var ctx = DeviceContext()
    print("Using GPU API:", ctx.api())

    var input_buffer_host = ctx.enqueue_create_host_buffer[DTYPE](total_floats)
    var input_buffer_device = ctx.enqueue_create_buffer[DTYPE](total_floats)
    var output_buffer_device = ctx.enqueue_create_buffer[DTYPE](total_floats)

    var input_image_host = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](
        input_buffer_host
    )
    var input_image_device = LayoutTensor[DTYPE, IMAGE_LAYOUT, ImmutAnyOrigin](
        input_buffer_device
    )
    var output_image_device = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](
        output_buffer_device
    )

    # Initialize input data (simplified ramp for testing)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            var r = Float32(x) / WIDTH
            var g = Float32(y) / HEIGHT
            var b = Float32(0.5)
            input_image_host.store[width=4](
                Index(y, x, 0), SIMD[DType.float32, 4](r, g, b, 1.0)
            )

    ctx.enqueue_copy(input_buffer_device, input_buffer_host)

    var white_target = Float32(1.0)
    var black_target = Float32(0.000152)
    var paper_exp = Float32(0.5)
    var film_fog = Float32(0.0)
    var contrast_power = Float32(2.5)
    var skew_power = Float32(1.0)
    var hue_preservation = Float32(1.0)

    var identity = SIMD[DType.float32, 16](0)
    identity[0] = 1; identity[5] = 1; identity[10] = 1; identity[15] = 1

    # Warmup
    run_sigmoid_rgb_ratio(ctx, output_image_device, input_image_device, white_target, black_target, paper_exp, film_fog, contrast_power, skew_power, WIDTH * HEIGHT)
    ctx.synchronize()

    # Benchmarking
    var bench = Bench(BenchConfig(max_iters=100, num_warmup_iters=10))

    @parameter
    fn bench_rgb(mut b: Bencher) raises:
        @parameter
        fn run(ctx: DeviceContext) raises:
            run_sigmoid_rgb_ratio(ctx, output_image_device, input_image_device, white_target, black_target, paper_exp, film_fog, contrast_power, skew_power, WIDTH * HEIGHT)
        b.iter_custom[run](ctx)
        ctx.synchronize()

    @parameter
    fn bench_per(mut b: Bencher) raises:
        @parameter
        fn run(ctx: DeviceContext) raises:
            run_sigmoid_per_channel(ctx, output_image_device, input_image_device, white_target, paper_exp, film_fog, contrast_power, skew_power, hue_preservation, identity, identity, identity, WIDTH * HEIGHT)
        b.iter_custom[run](ctx)
        ctx.synchronize()

    bench.bench_function[bench_rgb](BenchId("Mojo-Sigmoid-RGB-Ratio-New"))
    bench.bench_function[bench_per](BenchId("Mojo-Sigmoid-Per-Channel-New"))
    print(bench)
