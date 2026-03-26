from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from layout.runtime_layout import RuntimeLayout
from std.utils import Index, IndexList
from std.algorithm.functional import elementwise
from std.gpu.host.compile import get_gpu_target
from std.memory.unsafe_pointer import alloc, UnsafePointer
from std.sys import simd_width_of
from iop.sigmoid.kernels import (
    apply_sigmoid_rgb_ratio,
    apply_sigmoid_per_channel,
)

comptime CHANNELS = 4
comptime IMAGE_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, CHANNELS)
comptime DTYPE = DType.float32
comptime SIMD_WIDTH = simd_width_of[DTYPE, target=get_gpu_target()]()


struct MojoCtx:
    var use_gpu: Int
    var dctx_addr: Int

    fn __init__(out self, use_gpu: Int, dctx_addr: Int):
        self.use_gpu = use_gpu
        self.dctx_addr = dctx_addr


struct CParamsView:
    var p: UnsafePointer[Float32, MutAnyOrigin]

    fn __init__(out self, p: Int):
        self.p = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=p)

    # In SigmoidMojoParams, each scalar is a float[4] array
    fn white_target(self) -> Float32:
        return self.p[0]

    fn black_target(self) -> Float32:
        return self.p[4]

    fn paper_exposure(self) -> Float32:
        return self.p[8]

    fn film_fog(self) -> Float32:
        return self.p[12]

    fn film_power(self) -> Float32:
        return self.p[16]

    fn paper_power(self) -> Float32:
        return self.p[20]

    fn hue_preservation(self) -> Float32:
        return self.p[32]  # hue_preservation[4] at index 32

    fn pipe_to_base(self) -> SIMD[DType.float32, 16]:
        var m = SIMD[DType.float32, 16]()
        for i in range(16):
            m[i] = self.p[36 + i]
        return m

    fn base_to_rendering(self) -> SIMD[DType.float32, 16]:
        var m = SIMD[DType.float32, 16]()
        for i in range(16):
            m[i] = self.p[52 + i]
        return m

    fn rendering_to_pipe(self) -> SIMD[DType.float32, 16]:
        var m = SIMD[DType.float32, 16]()
        for i in range(16):
            m[i] = self.p[68 + i]
        return m


# =========================================================================
# INTERNAL GPU LAUNCHERS (ZERO-POINTER PARAMETERS)
# =========================================================================


fn _launch_rgb_ratio_gpu(
    dctx: DeviceContext,
    dev_in_ptr: Int,
    dev_out_ptr: Int,
    wt: Float32,
    bt: Float32,
    pe: Float32,
    ff: Float32,
    fp: Float32,
    pp: Float32,
    num_pixels: Int,
) raises:
    @parameter
    @always_inline
    fn gpu_kernel[
        sw: Int, rank: Int, align: Int
    ](indices: IndexList[rank]) capturing -> None:
        var idx = indices[0] * 4
        var pin = UnsafePointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=dev_in_ptr
        )
        var pout = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=dev_out_ptr
        )

        var r = apply_sigmoid_rgb_ratio(
            pin[idx],
            pin[idx + 1],
            pin[idx + 2],
            pin[idx + 3],
            wt,
            bt,
            pe,
            ff,
            fp,
            pp,
        )
        pout[idx] = r[0]
        pout[idx + 1] = r[1]
        pout[idx + 2] = r[2]
        pout[idx + 3] = r[3]

    elementwise[gpu_kernel, SIMD_WIDTH, target="gpu"](num_pixels, dctx)
    dctx.synchronize()


fn _launch_per_channel_gpu(
    dctx: DeviceContext,
    dev_in_ptr: Int,
    dev_out_ptr: Int,
    wt: Float32,
    pe: Float32,
    ff: Float32,
    fp: Float32,
    pp: Float32,
    hp: Float32,
    kptb: SIMD[DType.float32, 16],
    kbtr: SIMD[DType.float32, 16],
    krtp: SIMD[DType.float32, 16],
    num_pixels: Int,
) raises:
    @parameter
    @always_inline
    fn gpu_kernel[
        sw: Int, rank: Int, align: Int
    ](indices: IndexList[rank]) capturing -> None:
        var idx = indices[0] * 4
        var pin = UnsafePointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=dev_in_ptr
        )
        var pout = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=dev_out_ptr
        )

        var r = apply_sigmoid_per_channel(
            pin[idx],
            pin[idx + 1],
            pin[idx + 2],
            pin[idx + 3],
            wt,
            pe,
            ff,
            fp,
            pp,
            hp,
            kptb,
            kbtr,
            krtp,
        )
        pout[idx] = r[0]
        pout[idx + 1] = r[1]
        pout[idx + 2] = r[2]
        pout[idx + 3] = r[3]

    elementwise[gpu_kernel, SIMD_WIDTH, target="gpu"](num_pixels, dctx)
    dctx.synchronize()


# =========================================================================
# EXPORTED INTERFACE
# =========================================================================


@export
fn sigmoid_mojo_init(ctx_out: UnsafePointer[Int, MutAnyOrigin], use_gpu: Int32):
    var gpu = use_gpu != 0
    var dctx_addr = 0
    if gpu:
        try:
            var d_ptr = alloc[DeviceContext](1)
            d_ptr.init_pointee_move(DeviceContext())
            dctx_addr = Int(d_ptr)
            print("Mojo: GPU Context Initialized Successfully")
        except e:
            print("Mojo: GPU Init Error (falling back to CPU):", String(e))
            gpu = False
    else:
        print("Mojo: CPU Context Initialized")
    var p = alloc[MojoCtx](1)
    p[0].use_gpu = 1 if gpu else 0
    p[0].dctx_addr = dctx_addr
    ctx_out[0] = Int(p)


@export
fn sigmoid_mojo_destroy(ctx_addr: Int):
    var p = UnsafePointer[MojoCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)
    if p[0].dctx_addr != 0:
        var dctx_ptr = UnsafePointer[DeviceContext, MutAnyOrigin](
            unsafe_from_address=p[0].dctx_addr
        )
        dctx_ptr.destroy_pointee()
        dctx_ptr.free()
    p.free()


@export
fn sigmoid_mojo_rgb_ratio(
    ctx_addr: Int,
    in_addr: Int,
    out_addr: Int,
    width: Int32,
    height: Int32,
    p_addr: Int,
):
    var ctx_p = UnsafePointer[MojoCtx, MutAnyOrigin](
        unsafe_from_address=ctx_addr
    )
    var use_gpu = ctx_p[0].use_gpu != 0
    var dctx_addr = ctx_p[0].dctx_addr
    var params = CParamsView(p_addr)
    var in_p = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=in_addr)
    var out_p = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=out_addr
    )
    var h = Int(height)
    var w = Int(width)
    var num_pixels = h * w

    if use_gpu and dctx_addr != 0:
        try:
            var dctx = UnsafePointer[DeviceContext, MutAnyOrigin](
                unsafe_from_address=dctx_addr
            )[0]
            var dev_in = dctx.enqueue_create_buffer[DTYPE](num_pixels * 4)
            var dev_out = dctx.enqueue_create_buffer[DTYPE](num_pixels * 4)
            dctx.enqueue_copy(dev_in, in_p)

            _launch_rgb_ratio_gpu(
                dctx,
                Int(dev_in.unsafe_ptr()),
                Int(dev_out.unsafe_ptr()),
                params.white_target(),
                params.black_target(),
                params.paper_exposure(),
                params.film_fog(),
                params.film_power(),
                params.paper_power(),
                num_pixels,
            )

            dctx.enqueue_copy(out_p, dev_out)
            dctx.synchronize()
        except e:
            print("GPU Run Error (RGB Ratio):", String(e))
    else:
        var rt = RuntimeLayout[IMAGE_LAYOUT].row_major(
            IndexList[3](h, w, CHANNELS)
        )
        var in_t = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](in_p, rt)
        var out_t = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](out_p, rt)
        var wt = params.white_target()
        var bt = params.black_target()
        var pe = params.paper_exposure()
        var ff = params.film_fog()
        var fp = params.film_power()
        var pp = params.paper_power()
        for i in range(num_pixels):
            var px = in_t.load[width=4](Index(i // w, i % w, 0))
            var r = apply_sigmoid_rgb_ratio(
                px[0], px[1], px[2], px[3], wt, bt, pe, ff, fp, pp
            )
            out_t.store[width=4](Index(i // w, i % w, 0), r)


@export
fn sigmoid_mojo_per_channel(
    ctx_addr: Int,
    in_addr: Int,
    out_addr: Int,
    width: Int32,
    height: Int32,
    p_addr: Int,
):
    var ctx_p = UnsafePointer[MojoCtx, MutAnyOrigin](
        unsafe_from_address=ctx_addr
    )
    var use_gpu = ctx_p[0].use_gpu != 0
    var dctx_addr = ctx_p[0].dctx_addr
    var params = CParamsView(p_addr)
    var in_p = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=in_addr)
    var out_p = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=out_addr
    )
    var h = Int(height)
    var w = Int(width)
    var num_pixels = h * w

    if use_gpu and dctx_addr != 0:
        try:
            var dctx = UnsafePointer[DeviceContext, MutAnyOrigin](
                unsafe_from_address=dctx_addr
            )[0]
            var dev_in = dctx.enqueue_create_buffer[DTYPE](num_pixels * 4)
            var dev_out = dctx.enqueue_create_buffer[DTYPE](num_pixels * 4)
            dctx.enqueue_copy(dev_in, in_p)

            _launch_per_channel_gpu(
                dctx,
                Int(dev_in.unsafe_ptr()),
                Int(dev_out.unsafe_ptr()),
                params.white_target(),
                params.paper_exposure(),
                params.film_fog(),
                params.film_power(),
                params.paper_power(),
                params.hue_preservation(),
                params.pipe_to_base(),
                params.base_to_rendering(),
                params.rendering_to_pipe(),
                num_pixels,
            )

            dctx.enqueue_copy(out_p, dev_out)
            dctx.synchronize()
        except e:
            print("GPU Error (Per Channel):", String(e))
    else:
        var rt = RuntimeLayout[IMAGE_LAYOUT].row_major(
            IndexList[3](h, w, CHANNELS)
        )
        var in_t = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](in_p, rt)
        var out_t = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](out_p, rt)
        var wt = params.white_target()
        var pe = params.paper_exposure()
        var ff = params.film_fog()
        var fp = params.film_power()
        var pp = params.paper_power()
        var hp = params.hue_preservation()
        var ptb = params.pipe_to_base()
        var btr = params.base_to_rendering()
        var rtp = params.rendering_to_pipe()
        for i in range(num_pixels):
            var px = in_t.load[width=4](Index(i // w, i % w, 0))
            var r = apply_sigmoid_per_channel(
                px[0],
                px[1],
                px[2],
                px[3],
                wt,
                pe,
                ff,
                fp,
                pp,
                hp,
                ptb,
                btr,
                rtp,
            )
            out_t.store[width=4](Index(i // w, i % w, 0), r)
