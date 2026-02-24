from gpu import block_dim, block_idx, thread_idx
from gpu.host import DeviceContext, DeviceBuffer, Dim, HostBuffer
from layout import Layout, LayoutTensor
from utils import Index, IndexList
from math import ceildiv, isnan, sqrt, pow, max, min
from benchmark import Bench, BenchConfig, Bencher, BenchId
from algorithm.functional import elementwise

# Configuration
comptime WIDTH = 6016
comptime HEIGHT = 4016
comptime CHANNELS = 4
comptime IMAGE_LAYOUT = Layout.row_major(HEIGHT, WIDTH, CHANNELS)
comptime DTYPE = DType.float32


# Removed Pixel struct to use 3D LayoutTensor directly as requested


@always_inline
fn generalized_loglogistic_sigmoid_scalar(
    value: Float32,
    magnitude: Float32,
    paper_exp: Float32,
    film_fog: Float32,
    film_power: Float32,
    paper_power: Float32,
) -> Float32:
    var clamped_value = max(value, Float32(0.0))
    var film_response = pow(film_fog + clamped_value, film_power)
    var paper_response = magnitude * pow(
        film_response / (paper_exp + film_response), paper_power
    )
    if isnan(paper_response):
        return magnitude
    return paper_response


@always_inline
fn apply_sigmoid_rgb_ratio(
    in_r: Float32,
    in_g: Float32,
    in_b: Float32,
    in_a: Float32,
    white_target: Float32,
    black_target: Float32,
    paper_exp: Float32,
    film_fog: Float32,
    film_power: Float32,
    paper_power: Float32,
) -> SIMD[DType.float32, 4]:
    # Desaturate negative values
    var avg = max((in_r + in_g + in_b) / 3.0, Float32(0.0))
    var min_v = min(min(in_r, in_g), in_b)
    var sat = Float32(1.0)
    if min_v < 0.0:
        sat = -avg / (min_v - avg)

    var p_r = avg + sat * (in_r - avg)
    var p_g = avg + sat * (in_g - avg)
    var p_b = avg + sat * (in_b - avg)

    var luma = (p_r + p_g + p_b) / 3.0
    var mapped_luma = generalized_loglogistic_sigmoid_scalar(
        luma, white_target, paper_exp, film_fog, film_power, paper_power
    )

    if luma > 1e-9:
        var scale = mapped_luma / luma
        p_r *= scale
        p_g *= scale
        p_b *= scale
    else:
        p_r = mapped_luma
        p_g = mapped_luma
        p_b = mapped_luma

    var p_min = min(min(p_r, p_g), p_b)
    var p_max = max(max(p_r, p_g), p_b)
    var eps = Float32(1e-6)
    var d_white = (white_target - mapped_luma) / (p_max - mapped_luma + eps)
    var d_black = (black_target - mapped_luma) / (p_min - mapped_luma - eps)
    var db_vs_chroma = min(d_white, d_black)
    var cvm_border = (mapped_luma - p_min) / (mapped_luma + eps)
    var p_chr_adj = 1.0 / (cvm_border * db_vs_chroma + eps)
    var h_chr = (
        2.0 * cvm_border / (1.0 - cvm_border * cvm_border + eps)
    ) * p_chr_adj
    var h_z = sqrt(h_chr * h_chr + 1.0)
    var chroma_f = h_chr / (1.0 + h_z) * db_vs_chroma

    return SIMD[DType.float32, 4](
        mapped_luma + chroma_f * (p_r - mapped_luma),
        mapped_luma + chroma_f * (p_g - mapped_luma),
        mapped_luma + chroma_f * (p_b - mapped_luma),
        in_a,
    )


@always_inline
fn apply_sigmoid_per_channel(
    in_r: Float32,
    in_g: Float32,
    in_b: Float32,
    in_a: Float32,
    white_target: Float32,
    paper_exp: Float32,
    film_fog: Float32,
    contrast_power: Float32,
    skew_power: Float32,
    hue_preservation: Float32,
    pipe_to_base: SIMD[DType.float32, 16],
    base_to_rendering: SIMD[DType.float32, 16],
    rendering_to_pipe: SIMD[DType.float32, 16],
) -> SIMD[DType.float32, 4]:
    # 1. Transform to base space
    var i_r = (
        pipe_to_base[0] * in_r + pipe_to_base[1] * in_g + pipe_to_base[2] * in_b
    )
    var i_g = (
        pipe_to_base[4] * in_r + pipe_to_base[5] * in_g + pipe_to_base[6] * in_b
    )
    var i_b = (
        pipe_to_base[8] * in_r
        + pipe_to_base[9] * in_g
        + pipe_to_base[10] * in_b
    )

    # 2. Desaturate negative
    var avg = max((i_r + i_g + i_b) / 3.0, Float32(0.0))
    var min_v = min(min(i_r, i_g), i_b)
    var sat = Float32(1.0)
    if min_v < 0.0:
        sat = -avg / (min_v - avg)
    i_r = avg + sat * (i_r - avg)
    i_g = avg + sat * (i_g - avg)
    i_b = avg + sat * (i_b - avg)

    # 3. Transform to rendering space
    var r_r = (
        base_to_rendering[0] * i_r
        + base_to_rendering[1] * i_g
        + base_to_rendering[2] * i_b
    )
    var r_g = (
        base_to_rendering[4] * i_r
        + base_to_rendering[5] * i_g
        + base_to_rendering[6] * i_b
    )
    var r_b = (
        base_to_rendering[8] * i_r
        + base_to_rendering[9] * i_g
        + base_to_rendering[10] * i_b
    )

    # 4. Per-channel sigmoid curves
    var pc_r = generalized_loglogistic_sigmoid_scalar(
        r_r, white_target, paper_exp, film_fog, contrast_power, skew_power
    )
    var pc_g = generalized_loglogistic_sigmoid_scalar(
        r_g, white_target, paper_exp, film_fog, contrast_power, skew_power
    )
    var pc_b = generalized_loglogistic_sigmoid_scalar(
        r_b, white_target, paper_exp, film_fog, contrast_power, skew_power
    )

    # 5. Preserve hue & energy
    var p_min: Float32
    var p_mid: Float32
    var p_max: Float32
    var pc_min: Float32
    var pc_mid: Float32
    var pc_max: Float32

    if r_r >= r_g:
        if r_g >= r_b:  # R G B
            p_max = r_r
            p_mid = r_g
            p_min = r_b
            pc_max = pc_r
            pc_mid = pc_g
            pc_min = pc_b
        elif r_b >= r_r:  # B R G
            p_max = r_b
            p_mid = r_r
            p_min = r_g
            pc_max = pc_b
            pc_mid = pc_r
            pc_min = pc_g
        else:  # R B G
            p_max = r_r
            p_mid = r_b
            p_min = r_g
            pc_max = pc_r
            pc_mid = pc_b
            pc_min = pc_g
    else:
        if r_r >= r_b:  # G R B
            p_max = r_g
            p_mid = r_r
            p_min = r_b
            pc_max = pc_g
            pc_mid = pc_r
            pc_min = pc_b
        elif r_b >= r_g:  # B G R
            p_max = r_b
            p_mid = r_g
            p_min = r_r
            pc_max = pc_b
            pc_mid = pc_g
            pc_min = pc_r
        else:  # G B R
            p_max = r_g
            p_mid = r_b
            p_min = r_r
            pc_max = pc_g
            pc_mid = pc_b
            pc_min = pc_r

    var chroma = p_max - p_min
    var midscale = Float32(0.0)
    if chroma != 0.0:
        midscale = (p_mid - p_min) / chroma

    var f_hc = pc_min + (pc_max - pc_min) * midscale
    var n_mid = (1.0 - hue_preservation) * pc_mid + hue_preservation * f_hc

    var blend = 2.0 * p_min / (p_min + p_mid + 1e-9)
    var target = blend * (pc_r + pc_g + pc_b) + (1.0 - blend) * (
        pc_min + n_mid + pc_max
    )

    var res_min: Float32
    var res_mid: Float32
    var res_max: Float32
    if n_mid <= pc_mid:
        res_mid = (
            (1.0 - hue_preservation) * pc_mid
            + hue_preservation
            * (midscale * pc_max + (1.0 - midscale) * (target - pc_max))
        ) / (1.0 + hue_preservation * (1.0 - midscale))
        res_min = target - pc_max - res_mid
        res_max = pc_max
    else:
        res_mid = (
            (1.0 - hue_preservation) * pc_mid
            + hue_preservation
            * (pc_min * (1.0 - midscale) + midscale * (target - pc_min))
        ) / (1.0 + hue_preservation * midscale)
        res_min = pc_min
        res_max = target - pc_min - res_mid

    var res_r: Float32
    var res_g: Float32
    var res_b: Float32
    if r_r >= r_g:
        if r_g >= r_b:  # R G B
            res_r = res_max
            res_g = res_mid
            res_b = res_min
        elif r_b >= r_r:  # B R G
            res_b = res_max
            res_r = res_mid
            res_g = res_min
        else:  # R B G
            res_r = res_max
            res_b = res_mid
            res_g = res_min
    else:
        if r_r >= r_b:  # G R B
            res_g = res_max
            res_r = res_mid
            res_b = res_min
        elif r_b >= r_g:  # B G R
            res_b = res_max
            res_g = res_mid
            res_r = res_min
        else:  # G B R
            res_g = res_max
            res_b = res_mid
            res_r = res_min

    # 6. Transform to pipe space
    var out_r = (
        rendering_to_pipe[0] * res_r
        + rendering_to_pipe[1] * res_g
        + rendering_to_pipe[2] * res_b
    )
    var out_g = (
        rendering_to_pipe[4] * res_r
        + rendering_to_pipe[5] * res_g
        + rendering_to_pipe[6] * res_b
    )
    var out_b = (
        rendering_to_pipe[8] * res_r
        + rendering_to_pipe[9] * res_g
        + rendering_to_pipe[10] * res_b
    )

    return SIMD[DType.float32, 4](out_r, out_g, out_b, in_a)


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
            pix[0],
            pix[1],
            pix[2],
            pix[3],
            white_target,
            black_target,
            paper_exp,
            film_fog,
            film_power,
            paper_power,
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
            pix[0],
            pix[1],
            pix[2],
            pix[3],
            white_target,
            paper_exp,
            film_fog,
            contrast_power,
            skew_power,
            hue_preservation,
            pipe_to_base,
            base_to_rendering,
            rendering_to_pipe,
        )
        output.store[width=4](Index(y, x, 0), res)

    elementwise[per_channel_closure, 1, target="gpu"](num_pixels, ctx)


fn main() raises:
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

    for y in range(HEIGHT):
        var h = Float32(y) / (HEIGHT - 1) * 6.0
        var segment = Int(h)
        var f1 = h - Float32(segment)
        var pr: Float32
        var pg: Float32
        var pb: Float32
        if segment == 0:
            pr = 1.0
            pg = f1
            pb = 0.0
        elif segment == 1:
            pr = 1.0 - f1
            pg = 1.0
            pb = 0.0
        elif segment == 2:
            pr = 0.0
            pg = 1.0
            pb = f1
        elif segment == 3:
            pr = 0.0
            pg = 1.0 - f1
            pb = 1.0
        elif segment == 4:
            pr = f1
            pg = 0.0
            pb = 1.0
        elif segment == 5:
            pr = 1.0
            pg = 0.0
            pb = 1.0 - f1
        else:
            pr = 1.0
            pg = 0.0
            pb = 0.0

        for x in range(WIDTH):
            var r: Float32
            var g: Float32
            var b: Float32
            var mid_x = Float32(WIDTH) / 2.0
            if Float32(x) < mid_x:
                var t = Float32(x) / mid_x
                r = pr * t
                g = pg * t
                b = pb * t
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
    identity[0] = 1
    identity[5] = 1
    identity[10] = 1
    identity[15] = 1

    # Runs
    run_sigmoid_rgb_ratio(
        ctx,
        output_image_device,
        input_image_device,
        white_target,
        black_target,
        paper_exp,
        film_fog,
        contrast_power,
        skew_power,
        WIDTH * HEIGHT,
    )
    ctx.synchronize()

    var output_buffer_host = ctx.enqueue_create_host_buffer[DTYPE](total_floats)
    ctx.enqueue_copy(output_buffer_host, output_buffer_device)
    ctx.synchronize()
    var output_image_host = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](
        output_buffer_host
    )

    for i in range(1, 4):
        var t = Float32(i) * 0.2
        var x = Int(t * Float32(WIDTH))
        var y = Int(t * Float32(HEIGHT))
        var res = output_image_host.load[width=4](Index(y, x, 0))
        print(
            "RGB Ratio -",
            Int(t * 100),
            "% pixel: [",
            res[0],
            res[1],
            res[2],
            res[3],
            "]",
        )

    run_sigmoid_per_channel(
        ctx,
        output_image_device,
        input_image_device,
        white_target,
        paper_exp,
        film_fog,
        contrast_power,
        skew_power,
        hue_preservation,
        identity,
        identity,
        identity,
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
            "Per Channel -",
            Int(t * 100),
            "% pixel: [",
            res[0],
            res[1],
            res[2],
            res[3],
            "]",
        )

    # Benchmarking
    var bench = Bench(BenchConfig(max_iters=1000, num_warmup_iters=100))

    @parameter
    fn bench_rgb(mut b: Bencher) raises:
        @parameter
        fn run(ctx: DeviceContext) raises:
            run_sigmoid_rgb_ratio(
                ctx,
                output_image_device,
                input_image_device,
                white_target,
                black_target,
                paper_exp,
                film_fog,
                contrast_power,
                skew_power,
                WIDTH * HEIGHT,
            )

        b.iter_custom[run](ctx)
        ctx.synchronize()

    @parameter
    fn bench_per(mut b: Bencher) raises:
        @parameter
        fn run(ctx: DeviceContext) raises:
            run_sigmoid_per_channel(
                ctx,
                output_image_device,
                input_image_device,
                white_target,
                paper_exp,
                film_fog,
                contrast_power,
                skew_power,
                hue_preservation,
                identity,
                identity,
                identity,
                WIDTH * HEIGHT,
            )

        b.iter_custom[run](ctx)
        ctx.synchronize()

    bench.bench_function[bench_rgb](
        BenchId("Mojo-Sigmoid-GPU-RGB-Ratio-V2"), fixed_iterations=1000
    )
    bench.bench_function[bench_per](
        BenchId("Mojo-Sigmoid-GPU-Per-Channel-V2"), fixed_iterations=1000
    )
    print(bench)
