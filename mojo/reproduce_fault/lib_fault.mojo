from std.gpu.host import DeviceContext
from std.algorithm.functional import elementwise
from std.utils import IndexList
from std.memory.unsafe_pointer import alloc, UnsafePointer


fn internal_gpu_launcher(dctx: DeviceContext, p_src: Int, p_dst: Int) raises:
    @parameter
    @always_inline
    fn kernel[
        sw: Int, rank: Int, align: Int
    ](indices: IndexList[rank]) capturing -> None:
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=p_dst)[
            0
        ] = UnsafePointer[Float32, ImmutAnyOrigin](unsafe_from_address=p_src)[0]

    elementwise[kernel, 1, target="cpu"](1, dctx)
    dctx.synchronize()


@export
fn run_working_case():
    try:
        print("Mojo: [WORKING] Launching via internal_gpu_launcher...")
        var dctx = DeviceContext()
        var dev_data = dctx.enqueue_create_buffer[DType.float32](1)
        var dev_out = dctx.enqueue_create_buffer[DType.float32](1)

        var p_src = Int(dev_data.unsafe_ptr())
        var p_dst = Int(dev_out.unsafe_ptr())

        internal_gpu_launcher(dctx, p_src, p_dst)
        print("Mojo: [WORKING] Success.")
    except e:
        print("Mojo: [WORKING] Error:", String(e))


@export
fn run_failing_case():
    try:
        print(
            "Mojo: [FAILING] Direct launch from @export (SIGSEGV/Fault"
            " expected)..."
        )
        var dctx = DeviceContext()
        var dev_data = dctx.enqueue_create_buffer[DType.float32](1)
        var dev_out = dctx.enqueue_create_buffer[DType.float32](1)

        var p_src = Int(dev_data.unsafe_ptr())
        var p_dst = Int(dev_out.unsafe_ptr())

        @parameter
        @always_inline
        fn kernel[
            sw: Int, rank: Int, align: Int
        ](indices: IndexList[rank]) capturing -> None:
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=p_dst)[
                0
            ] = UnsafePointer[Float32, ImmutAnyOrigin](
                unsafe_from_address=p_src
            )[
                0
            ]

        elementwise[kernel, 1, target="cpu"](1, dctx)
        dctx.synchronize()
        print("Mojo: [FAILING] Success (Unexpected!)")
    except e:
        print("Mojo: [FAILING] Error:", String(e))
