from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from layout.runtime_layout import RuntimeLayout, IndexList
from layout.coord import Coord
from std.utils import Index
from std.algorithm.functional import elementwise, vectorize
from std.gpu.host.compile import get_gpu_target
from std.memory import alloc
from std.memory.unsafe_pointer import UnsafePointer
from std.sys import simd_width_of
from iop.blurs.kernels import apply_convolve, apply_convolve_vector

comptime CHANNELS = 4
comptime IMAGE_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, CHANNELS)
comptime DTYPE = DType.float32
comptime SIMD_WIDTH = simd_width_of[DTYPE, target=get_gpu_target()]()


struct MojoCtx:
    var use_gpu: Int
    var dctx_addr: Int

    def __init__(out self, use_gpu: Int, dctx_addr: Int):
        self.use_gpu = use_gpu
        self.dctx_addr = dctx_addr


def _launch_convolve_gpu(
    dctx: DeviceContext,
    in_t: LayoutTensor[DTYPE, IMAGE_LAYOUT, ImmutAnyOrigin],
    out_t: LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin],
    kern_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    width: Int,
    height: Int,
    radius: Int,
    num_pixels: Int,
) raises:
    @parameter
    @always_inline
    def gpu_kernel[simd_width: Int, alignment: Int](coord: Coord) capturing -> None:
        var px_idx = Int(coord[0].value())
        var y = px_idx // width
        var x = px_idx % width
        var res = apply_convolve_vector[1](
            x, y, width, height, radius, in_t, kern_ptr
        )
        out_t.store[width=4](Index(y, x, 0), res)

    elementwise[gpu_kernel, 1, target="gpu"](num_pixels, dctx)
    dctx.synchronize()


@export("blurs_mojo_init")
def blurs_mojo_init(
    ctx_out: UnsafePointer[Int, MutAnyOrigin], use_gpu: Int32
) abi("C") -> None:
    var gpu = use_gpu != 0
    var dctx_addr = 0
    if gpu:
        try:
            var d_ptr = alloc[DeviceContext](1)
            d_ptr.unsafe_write(DeviceContext())
            dctx_addr = Int(d_ptr)
            print("Mojo Blurs: GPU Context Initialized")
        except e:
            print("Mojo Blurs: GPU Init Error:", String(e))
            gpu = False
    else:
        print("Mojo Blurs: CPU Context Initialized")
    var p = alloc[MojoCtx](1)
    p[0].use_gpu = 1 if gpu else 0
    p[0].dctx_addr = dctx_addr
    ctx_out[0] = Int(p)


@export("blurs_mojo_destroy")
def blurs_mojo_destroy(ctx_addr: Int) abi("C") -> None:
    var p = UnsafePointer[MojoCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)
    if p[0].dctx_addr != 0:
        var dctx_ptr = UnsafePointer[DeviceContext, MutAnyOrigin](
            unsafe_from_address=p[0].dctx_addr
        )
        dctx_ptr.unsafe_deinit_pointee()
        dctx_ptr.free()
    p.free()


@export("blurs_mojo_convolve")
def blurs_mojo_convolve(
    ctx_addr: Int,
    in_addr: Int,
    kern_addr: Int,
    out_addr: Int,
    width: Int32,
    height: Int32,
    radius: Int32,
) abi("C") -> None:
    var ctx_p = UnsafePointer[MojoCtx, MutAnyOrigin](
        unsafe_from_address=ctx_addr
    )
    var use_gpu = ctx_p[0].use_gpu != 0
    var dctx_addr = ctx_p[0].dctx_addr

    var kern_p = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=kern_addr
    )
    var out_p = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=out_addr
    )

    var h = Int(height)
    var w = Int(width)
    var r = Int(radius)
    var num_pixels = h * w
    var k_size = (2 * r + 1) * (2 * r + 1)

    var rt = RuntimeLayout[IMAGE_LAYOUT].row_major(
        IndexList[3](h, w, CHANNELS)
    )

    if use_gpu and dctx_addr != 0:
        try:
            var in_p = UnsafePointer[Float32, MutAnyOrigin](
                unsafe_from_address=in_addr
            )
            var dctx = UnsafePointer[DeviceContext, MutAnyOrigin](
                unsafe_from_address=dctx_addr
            )[0]
            var dev_in = dctx.enqueue_create_buffer[DTYPE](num_pixels * 4)
            var dev_kern = dctx.enqueue_create_buffer[DTYPE](k_size)
            var dev_out = dctx.enqueue_create_buffer[DTYPE](num_pixels * 4)

            dctx.enqueue_copy(dev_in, in_p)
            dctx.enqueue_copy(dev_kern, kern_p)

            var in_t = LayoutTensor[DTYPE, IMAGE_LAYOUT, ImmutAnyOrigin](
                UnsafePointer[Float32, ImmutAnyOrigin](
                    unsafe_from_address=Int(dev_in.unsafe_ptr())
                ),
                rt,
            )
            var out_t = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](
                UnsafePointer[Float32, MutAnyOrigin](
                    unsafe_from_address=Int(dev_out.unsafe_ptr())
                ),
                rt,
            )
            var kern_ptr_gpu = UnsafePointer[Float32, ImmutAnyOrigin](
                unsafe_from_address=Int(dev_kern.unsafe_ptr())
            )

            _launch_convolve_gpu(
                dctx, in_t, out_t, kern_ptr_gpu, w, h, r, num_pixels,
            )

            dctx.enqueue_copy(out_p, dev_out)
            dctx.synchronize()
        except e:
            print("GPU Run Error (Convolve):", String(e))
    else:
        var in_t = LayoutTensor[DTYPE, IMAGE_LAYOUT, ImmutAnyOrigin](
            UnsafePointer[Float32, ImmutAnyOrigin](
                unsafe_from_address=in_addr
            ),
            rt,
        )
        var out_t = LayoutTensor[DTYPE, IMAGE_LAYOUT, MutAnyOrigin](
            out_p,
            rt,
        )
        var kern_ptr_cpu = UnsafePointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=kern_addr
        )

        for y in range(h):
            var cur_y = y
            def row_fn[width: Int](x: Int) {mut}:
                var res = apply_convolve_vector[width](
                    x, cur_y, w, h, r, in_t, kern_ptr_cpu
                )
                for i in range(width):
                    var pixel = SIMD[DType.float32, 4](
                        res[i * 4],
                        res[i * 4 + 1],
                        res[i * 4 + 2],
                        res[i * 4 + 3],
                    )
                    out_t.store[width=4](Index(cur_y, x + i, 0), pixel)

            vectorize[SIMD_WIDTH](w, row_fn)
