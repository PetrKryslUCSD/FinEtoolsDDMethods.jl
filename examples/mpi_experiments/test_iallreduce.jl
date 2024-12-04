# include("common.jl")

using Test
using MPI

RBuffer = MPI.RBuffer
Op = MPI.Op
API  = MPI.API
MPI_Op = MPI.MPI_Op
IN_PLACE = MPI.IN_PLACE
Comm = MPI.Comm
AbstractRequest = MPI.AbstractRequest
Request = MPI.Request
_doc_external = x -> "For more information, see the [MPI documentation]($x)."

## Iallreduce

# mutating
"""
    Iallreduce!(sendbuf, recvbuf, op, comm::Comm, req::AbstractRequest=Request())
    Iallreduce!(sendrecvbuf, op, comm::Comm, req::AbstractRequest=Request())

Performs elementwise reduction using the operator `op` on the buffer `sendbuf`, storing
the result in the `recvbuf` of all processes in the group.

If only one `sendrecvbuf` buffer is provided, then the operation is performed in-place.

# See also
- [`Iallreduce`](@ref), to handle allocation of the output buffer.
- [`Op`](@ref) for details on reduction operators.

# External links
$(_doc_external("MPI_Iallreduce"))
"""
function Iallreduce!(rbuf::RBuffer, op::Union{Op,MPI_Op}, comm::Comm, req::AbstractRequest=Request())
    # int MPI_Iallreduce(const void *sendbuf, void *recvbuf, int count,
    # MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
    # MPI_Request *request)
    API.MPI_Iallreduce(rbuf.senddata, rbuf.recvdata, rbuf.count, rbuf.datatype, op, comm, req)
    return req
end
Iallreduce!(rbuf::RBuffer, op, comm::Comm, req::AbstractRequest=Request()) =
    Iallreduce!(rbuf, Op(op, eltype(rbuf)), comm, req)
Iallreduce!(sendbuf, recvbuf, op, comm::Comm, req::AbstractRequest=Request()) =
    Iallreduce!(RBuffer(sendbuf, recvbuf), op, comm, req)

# inplace
Iallreduce!(buf, op, comm::Comm, req::AbstractRequest=Request()) = Iallreduce!(IN_PLACE, buf, op, comm, req)


if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    import CUDA
    ArrayType = CUDA.CuArray
    synchronize() = CUDA.synchronize()
elseif get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "ROCArray"
    import AMDGPU
    ArrayType = AMDGPU.ROCArray
    synchronize() = AMDGPU.synchronize()
else
    ArrayType = Array
    synchronize() = nothing
end

# those are the tested MPI types, don't remove !
const MPITestTypes = [
    Char,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64, ComplexF32, ComplexF64
]

MPI.Init()

comm_size = MPI.Comm_size(MPI.COMM_WORLD)

if ArrayType != Array ||
   MPI.MPI_LIBRARY == "MicrosoftMPI" && Sys.WORD_SIZE == 32 ||
   Sys.ARCH === :powerpc64le || Sys.ARCH === :ppc64le ||
   Sys.ARCH === :aarch64 || startswith(string(Sys.ARCH), "arm")
    operators = [MPI.SUM, +]
else
    operators = [MPI.SUM, +, (x,y) -> 2x+y-x]
end

for T = [Int]
    for dims = [1, 2, 3]
        send_arr = ArrayType(zeros(T, Tuple(3 for i in 1:dims)))
        send_arr[:] .= 1:length(send_arr)
        synchronize()

        for op in operators

            # Non allocating version
            recv_arr = ArrayType{T}(undef, size(send_arr))
            req = Iallreduce!(send_arr, recv_arr, op, MPI.COMM_WORLD)
            sleep(rand())
            MPI.Wait(req)
            @test recv_arr == comm_size .* send_arr

            # Assertions when output buffer too small
            recv_arr = ArrayType{T}(undef, size(send_arr).-1)
            @test_throws AssertionError Iallreduce!(send_arr, recv_arr,
                                                       op, MPI.COMM_WORLD)
            # IN_PLACE
            recv_arr = copy(send_arr)
            synchronize()
            req = Iallreduce!(recv_arr, op, MPI.COMM_WORLD)
            sleep(rand())
            MPI.Wait(req)
            @test recv_arr == comm_size .* send_arr
        end
    end
end


MPI.Barrier( MPI.COMM_WORLD )

GC.gc()
MPI.Finalize()
@test MPI.Finalized()
