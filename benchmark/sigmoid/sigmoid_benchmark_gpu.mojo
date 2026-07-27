from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.utils import Index
from std.math import sqrt
from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from layout.runtime_layout import RuntimeLayout, IndexList
from std.memory.unsafe_pointer import UnsafePointer
from iop.sigmoid.kernels import (
    apply_sigmoid_rgb_ratio,
    apply_sigmoid_per_channel,
)
from iop.sigmoid.lib import (
    _launch_rgb_ratio_gpu,
    _launch_per_channel_gpu,
)

comptime WIDTH = 6016
comptime HEIGHT = 4016
comptime CHANNELS = 4
comptime IMAGE_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, CHANNELS)
comptime DTYPE = DType.float32


def main() raises:
    var total_floats = HEIGHT * WIDTH * CHANNELS
    var ctx = DeviceContext()
    print("Using GPU API:", ctx.api())

    var input_buffer_host = ctx.enqueue_create_host_buffer[DTYPE](total_floats)
    var input_buffer_device = ctx.enqueue_create_buffer[DTYPE](total_floats)
    var output_buffer_device = ctx.enqueue_create_buffer[DTYPE](total_floats)

    var rt = RuntimeLayout[IMAGE_LAYOUT].row_major(
        IndexList[3](HEIGHT, WIDTH, CHANNELS)
    )

    var input_image_host = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](
        UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=Int(input_buffer_host.unsafe_ptr())
        ),
        rt,
    )
    var input_image_device = LayoutTensor[DTYPE, IMAGE_LAYOUT, ImmutAnyOrigin](
        UnsafePointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(input_buffer_device.unsafe_ptr())
        ),
        rt,
    )
    var output_image_device = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](
        UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=Int(output_buffer_device.unsafe_ptr())
        ),
        rt,
    )

    for y in range(HEIGHT):
        var h = Float32(y) / (HEIGHT - 1) * 6.0
        var segment = Int(h)
        var f1 = h - Float32(segment)
        var pr: Float32
        var pg: Float32
        var pb: Float32
        if segment == 0:
            pr = 1.0; pg = f1; pb = 0.0
        elif segment == 1:
            pr = 1.0 - f1; pg = 1.0; pb = 0.0
        elif segment == 2:
            pr = 0.0; pg = 1.0; pb = f1
        elif segment == 3:
            pr = 0.0; pg = 1.0 - f1; pb = 1.0
        elif segment == 4:
            pr = f1; pg = 0.0; pb = 1.0
        elif segment == 5:
            pr = 1.0; pg = 0.0; pb = 1.0 - f1
        else:
            pr = 1.0; pg = 0.0; pb = 0.0

        for x in range(WIDTH):
            var r: Float32
            var g: Float32
            var b: Float32
            var mid_x = Float32(WIDTH) / 2.0
            if Float32(x) < mid_x:
                var t = Float32(x) / mid_x
                r = pr * t; g = pg * t; b = pb * t
            else:
                var t = (Float32(x) - mid_x) / (Float32(WIDTH) - 1.0 - mid_x)
                r = pr * (1.0 - t) + t
                g = pg * (1.0 - t) + t
                b = pb * (1.0 - t) + t
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

    _launch_rgb_ratio_gpu(
        ctx,
        input_image_device,
        output_image_device,
        white_target,
        black_target,
        paper_exp,
        film_fog,
        contrast_power,
        skew_power,
        WIDTH,
        HEIGHT,
        WIDTH * HEIGHT,
    )
    ctx.synchronize()

    var output_buffer_host = ctx.enqueue_create_host_buffer[DTYPE](total_floats)
    ctx.enqueue_copy(output_buffer_host, output_buffer_device)
    ctx.synchronize()
    var output_image_host = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](
        UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=Int(output_buffer_host.unsafe_ptr())
        ),
        rt,
    )

    for i in range(1, 4):
        var t = Float32(i) * 0.2
        var x = Int(t * Float32(WIDTH))
        var y = Int(t * Float32(HEIGHT))
        var res = output_image_host.load[width=4](Index(y, x, 0))
        print(
            "RGB Ratio -", Int(t * 100), "% pixel: [",
            res[0], res[1], res[2], res[3], "]",
        )

    _launch_per_channel_gpu(
        ctx,
        input_image_device,
        output_image_device,
        white_target,
        paper_exp,
        film_fog,
        contrast_power,
        skew_power,
        hue_preservation,
        identity,
        identity,
        identity,
        WIDTH,
        HEIGHT,
        WIDTH * HEIGHT,
    )
    ctx.synchronize()

    ctx.enqueue_copy(output_buffer_host, output_buffer_device)
    ctx.synchronize()

    for i in range(1, 4):
        var t = Float32(i) * 0.2
        var x = Int(t * Float32(WIDTH))
        var y = Int(t * Float32(HEIGHT))
        var res = output_image_host.load[width=4](Index(y, x, 0))
        print(
            "Per Channel -", Int(t * 100), "% pixel: [",
            res[0], res[1], res[2], res[3], "]",
        )

    var bench = Bench(BenchConfig(
        max_iters=200, num_warmup_iters=20, max_batch_size=20,
    ))

    @parameter
    def bench_rgb(mut b: Bencher) raises:
        @parameter
        def run_rgb(ctx: DeviceContext) raises:
            _launch_rgb_ratio_gpu(
                ctx,
                input_image_device,
                output_image_device,
                white_target,
                black_target,
                paper_exp,
                film_fog,
                contrast_power,
                skew_power,
                WIDTH,
                HEIGHT,
                WIDTH * HEIGHT,
            )

        b.iter_custom[run_rgb](ctx)
        ctx.synchronize()

    @parameter
    def bench_per(mut b: Bencher) raises:
        @parameter
        def run_per(ctx: DeviceContext) raises:
            _launch_per_channel_gpu(
                ctx,
                input_image_device,
                output_image_device,
                white_target,
                paper_exp,
                film_fog,
                contrast_power,
                skew_power,
                hue_preservation,
                identity,
                identity,
                identity,
                WIDTH,
                HEIGHT,
                WIDTH * HEIGHT,
            )

        b.iter_custom[run_per](ctx)
        ctx.synchronize()

    bench.bench_function[bench_rgb](
        BenchId("Mojo-Sigmoid-GPU-RGB-Ratio-V2"),
    )
    bench.bench_function[bench_per](
        BenchId("Mojo-Sigmoid-GPU-Per-Channel-V2"),
    )
    print(bench)

    for idx in range(len(bench.info_vec)):
        ref info = bench.info_vec[idx]
        var overall_mean = info.result.mean("ms")
        var sum_sq = 0.0
        var v_n = Float64(len(info.result.runs))
        for b in range(len(info.result.runs)):
            var batch_mean = info.result.runs[b].mean("ms")
            var d = batch_mean - overall_mean
            sum_sq += d * d
        var variance = sum_sq / v_n
        var std = sqrt(variance)
        print(info.name, "- std:", std, "ms")
