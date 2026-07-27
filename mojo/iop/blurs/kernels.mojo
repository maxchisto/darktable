from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from std.math import clamp
from std.utils import Index
from std.memory.unsafe_pointer import UnsafePointer

comptime CHANNELS = 4
comptime IMAGE_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, CHANNELS)
comptime DTYPE = DType.float32


@always_inline
def apply_convolve_vector[W: Int](
    x: Int,
    y: Int,
    width: Int,
    height: Int,
    radius: Int,
    in_t: LayoutTensor[DTYPE, IMAGE_LAYOUT, ImmutAnyOrigin],
    kern_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
) -> SIMD[DType.float32, W * 4]:
    var acc = SIMD[DType.float32, W * 4](0)

    for l in range(-radius, radius + 1):
        var ii = clamp(y + l, 0, height - 1)
        for m in range(-radius, radius + 1):
            var ik = l + radius
            var jk = m + radius
            var k_width = 2 * radius + 1
            var k = kern_ptr[ik * k_width + jk]

            for i in range(W):
                var jj = clamp(x + i + m, 0, width - 1)
                var pix = in_t.load[width=4](Index(ii, jj, 0))

                acc[i * 4] += k * pix[0]
                acc[i * 4 + 1] += k * pix[1]
                acc[i * 4 + 2] += k * pix[2]
                acc[i * 4 + 3] += k * pix[3]

    for i in range(W):
        var curr_x = x + i
        if curr_x < width:
            var orig = in_t.load[width=4](Index(y, curr_x, 0))
            acc[i * 4 + 3] = orig[3]

    return acc


@always_inline
def apply_convolve(
    x: Int,
    y: Int,
    width: Int,
    height: Int,
    radius: Int,
    in_t: LayoutTensor[DTYPE, IMAGE_LAYOUT, ImmutAnyOrigin],
    kern_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
) -> SIMD[DType.float32, 4]:
    return apply_convolve_vector[1](x, y, width, height, radius, in_t, kern_ptr)
