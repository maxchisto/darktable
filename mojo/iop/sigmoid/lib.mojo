from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from layout.runtime_layout import RuntimeLayout, IndexList
from layout.coord import Coord
from std.utils import Index
from std.algorithm.functional import elementwise
from std.memory import alloc
from std.memory.unsafe_pointer import UnsafePointer
from iop.sigmoid.kernels import (
    apply_sigmoid_rgb_ratio,
    apply_sigmoid_per_channel,
)

comptime CHANNELS = 4
comptime IMAGE_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, CHANNELS)
comptime DTYPE = DType.float32


struct MojoCtx:
    var use_gpu: Int
    var dctx_addr: Int

    def __init__(out self, use_gpu: Int, dctx_addr: Int):
        self.use_gpu = use_gpu
        self.dctx_addr = dctx_addr


struct CParams:
    var wt: Float32
    var bt: Float32
    var pe: Float32
    var ff: Float32
    var fp: Float32
    var pp: Float32
    var hp: Float32
    var ptb: SIMD[DType.float32, 16]
    var btr: SIMD[DType.float32, 16]
    var rtp: SIMD[DType.float32, 16]

    def __init__(out self, addr: Int):
        var p = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=addr)
        self.wt = p[0]
        self.bt = p[4]
        self.pe = p[8]
        self.ff = p[12]
        self.fp = p[16]
        self.pp = p[20]
        self.hp = p[32]
        var m_ptb = SIMD[DType.float32, 16]()
        var m_btr = SIMD[DType.float32, 16]()
        var m_rtp = SIMD[DType.float32, 16]()
        for i in range(16):
            m_ptb[i] = p[36 + i]
            m_btr[i] = p[52 + i]
            m_rtp[i] = p[68 + i]
        self.ptb = m_ptb
        self.btr = m_btr
        self.rtp = m_rtp


def _launch_rgb_ratio_gpu(
    dctx: DeviceContext,
    in_t: LayoutTensor[DTYPE, IMAGE_LAYOUT, ImmutAnyOrigin],
    out_t: LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin],
    wt: Float32,
    bt: Float32,
    pe: Float32,
    ff: Float32,
    fp: Float32,
    pp: Float32,
    img_w: Int,
    img_h: Int,
    num_pixels: Int,
) raises:
    @parameter
    @always_inline
    def gpu_kernel[simd_width: Int, alignment: Int](coord: Coord) capturing -> None:
        var px_idx = Int(coord[0].value())
        var y = px_idx // img_w
        var x = px_idx % img_w
        var pix = in_t.load[width=4](Index(y, x, 0))
        var r = apply_sigmoid_rgb_ratio(
            pix[0],
            pix[1],
            pix[2],
            pix[3],
            wt,
            bt,
            pe,
            ff,
            fp,
            pp,
        )
        out_t.store[width=4](Index(y, x, 0), r)

    elementwise[gpu_kernel, 1, target="gpu"](num_pixels, dctx)
    dctx.synchronize()


def _launch_per_channel_gpu(
    dctx: DeviceContext,
    in_t: LayoutTensor[DTYPE, IMAGE_LAYOUT, ImmutAnyOrigin],
    out_t: LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin],
    wt: Float32,
    pe: Float32,
    ff: Float32,
    fp: Float32,
    pp: Float32,
    hp: Float32,
    kptb: SIMD[DType.float32, 16],
    kbtr: SIMD[DType.float32, 16],
    krtp: SIMD[DType.float32, 16],
    img_w: Int,
    img_h: Int,
    num_pixels: Int,
) raises:
    @parameter
    @always_inline
    def gpu_kernel[simd_width: Int, alignment: Int](coord: Coord) capturing -> None:
        var px_idx = Int(coord[0].value())
        var y = px_idx // img_w
        var x = px_idx % img_w
        var pix = in_t.load[width=4](Index(y, x, 0))
        var r = apply_sigmoid_per_channel(
            pix[0],
            pix[1],
            pix[2],
            pix[3],
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
        out_t.store[width=4](Index(y, x, 0), r)

    elementwise[gpu_kernel, 1, target="gpu"](num_pixels, dctx)
    dctx.synchronize()


@export("sigmoid_mojo_init")
def sigmoid_mojo_init(ctx_out: UnsafePointer[Int, MutAnyOrigin], use_gpu: Int32) abi("C") -> None:
    var gpu = use_gpu != 0
    var dctx_addr = 0
    if gpu:
        try:
            var d_ptr = alloc[DeviceContext](1)
            d_ptr.unsafe_write(DeviceContext())
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


@export("sigmoid_mojo_destroy")
def sigmoid_mojo_destroy(ctx_addr: Int) abi("C") -> None:
    var p = UnsafePointer[MojoCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)
    if p[0].dctx_addr != 0:
        var dctx_ptr = UnsafePointer[DeviceContext, MutAnyOrigin](
            unsafe_from_address=p[0].dctx_addr
        )
        dctx_ptr.unsafe_deinit_pointee()
        dctx_ptr.free()
    p.free()


def _run_cpu_rgb_ratio(
    in_t: LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin],
    out_t: LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin],
    wt: Float32,
    bt: Float32,
    pe: Float32,
    ff: Float32,
    fp: Float32,
    pp: Float32,
    width: Int,
    height: Int,
    num_pixels: Int,
):
    for i in range(num_pixels):
        var px = in_t.load[width=4](Index(i // width, i % width, 0))
        var r = apply_sigmoid_rgb_ratio(
            px[0], px[1], px[2], px[3], wt, bt, pe, ff, fp, pp
        )
        out_t.store[width=4](Index(i // width, i % width, 0), r)


def _run_cpu_per_channel(
    in_t: LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin],
    out_t: LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin],
    wt: Float32,
    pe: Float32,
    ff: Float32,
    fp: Float32,
    pp: Float32,
    hp: Float32,
    ptb: SIMD[DType.float32, 16],
    btr: SIMD[DType.float32, 16],
    rtp: SIMD[DType.float32, 16],
    width: Int,
    height: Int,
    num_pixels: Int,
):
    for i in range(num_pixels):
        var px = in_t.load[width=4](Index(i // width, i % width, 0))
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
        out_t.store[width=4](Index(i // width, i % width, 0), r)


@export("sigmoid_mojo_rgb_ratio")
def sigmoid_mojo_rgb_ratio(
    ctx_addr: Int,
    in_addr: Int,
    out_addr: Int,
    width: Int32,
    height: Int32,
    p_addr: Int,
) abi("C") -> None:
    var ctx_p = UnsafePointer[MojoCtx, MutAnyOrigin](
        unsafe_from_address=ctx_addr
    )
    var use_gpu = ctx_p[0].use_gpu != 0
    var dctx_addr = ctx_p[0].dctx_addr
    var params = CParams(p_addr)
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
            var in_p = UnsafePointer[Float32, MutAnyOrigin](
                unsafe_from_address=in_addr
            )
            dctx.enqueue_copy(dev_in, in_p)

            var rt_gpu = RuntimeLayout[IMAGE_LAYOUT].row_major(
                IndexList[3](h, w, CHANNELS)
            )
            var in_t = LayoutTensor[DTYPE, IMAGE_LAYOUT, ImmutAnyOrigin](
                UnsafePointer[Float32, ImmutAnyOrigin](
                    unsafe_from_address=Int(dev_in.unsafe_ptr())
                ),
                rt_gpu,
            )
            var out_t = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](
                UnsafePointer[Float32, MutAnyOrigin](
                    unsafe_from_address=Int(dev_out.unsafe_ptr())
                ),
                rt_gpu,
            )

            _launch_rgb_ratio_gpu(
                dctx,
                in_t,
                out_t,
                params.wt,
                params.bt,
                params.pe,
                params.ff,
                params.fp,
                params.pp,
                w,
                h,
                num_pixels,
            )

            var out_p = UnsafePointer[Float32, MutAnyOrigin](
                unsafe_from_address=out_addr
            )
            dctx.enqueue_copy(out_p, dev_out)
            dctx.synchronize()
        except e:
            print("GPU Run Error (RGB Ratio):", String(e))
    else:
        var rt = RuntimeLayout[IMAGE_LAYOUT].row_major(
            IndexList[3](h, w, CHANNELS)
        )
        var in_p = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=in_addr
        )
        var out_p = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=out_addr
        )
        var in_t = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](in_p, rt)
        var out_t = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](out_p, rt)

        _run_cpu_rgb_ratio(
            in_t,
            out_t,
            params.wt,
            params.bt,
            params.pe,
            params.ff,
            params.fp,
            params.pp,
            w,
            h,
            num_pixels,
        )


@export("sigmoid_mojo_per_channel")
def sigmoid_mojo_per_channel(
    ctx_addr: Int,
    in_addr: Int,
    out_addr: Int,
    width: Int32,
    height: Int32,
    p_addr: Int,
) abi("C") -> None:
    var ctx_p = UnsafePointer[MojoCtx, MutAnyOrigin](
        unsafe_from_address=ctx_addr
    )
    var use_gpu = ctx_p[0].use_gpu != 0
    var dctx_addr = ctx_p[0].dctx_addr
    var params = CParams(p_addr)
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
            var in_p = UnsafePointer[Float32, MutAnyOrigin](
                unsafe_from_address=in_addr
            )
            dctx.enqueue_copy(dev_in, in_p)

            var rt_gpu = RuntimeLayout[IMAGE_LAYOUT].row_major(
                IndexList[3](h, w, CHANNELS)
            )
            var in_t = LayoutTensor[DTYPE, IMAGE_LAYOUT, ImmutAnyOrigin](
                UnsafePointer[Float32, ImmutAnyOrigin](
                    unsafe_from_address=Int(dev_in.unsafe_ptr())
                ),
                rt_gpu,
            )
            var out_t = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](
                UnsafePointer[Float32, MutAnyOrigin](
                    unsafe_from_address=Int(dev_out.unsafe_ptr())
                ),
                rt_gpu,
            )

            _launch_per_channel_gpu(
                dctx,
                in_t,
                out_t,
                params.wt,
                params.pe,
                params.ff,
                params.fp,
                params.pp,
                params.hp,
                params.ptb,
                params.btr,
                params.rtp,
                w,
                h,
                num_pixels,
            )

            var out_p = UnsafePointer[Float32, MutAnyOrigin](
                unsafe_from_address=out_addr
            )
            dctx.enqueue_copy(out_p, dev_out)
            dctx.synchronize()
        except e:
            print("GPU Error (Per Channel):", String(e))
    else:
        var rt = RuntimeLayout[IMAGE_LAYOUT].row_major(
            IndexList[3](h, w, CHANNELS)
        )
        var in_p = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=in_addr
        )
        var out_p = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=out_addr
        )
        var in_t = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](in_p, rt)
        var out_t = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](out_p, rt)

        _run_cpu_per_channel(
            in_t,
            out_t,
            params.wt,
            params.pe,
            params.ff,
            params.fp,
            params.pp,
            params.hp,
            params.ptb,
            params.btr,
            params.rtp,
            w,
            h,
            num_pixels,
        )
