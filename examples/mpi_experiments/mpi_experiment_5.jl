# examples/03-reduce.jl
# This example shows how to use custom datatypes and reduction operators
# It computes the variance in parallel in a numerically stable way

using MPI, Statistics

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


MPI.Init()
const comm = MPI.COMM_WORLD
const root = 0

rank = MPI.Comm_rank(comm)

X = fill(rank, 7)

# Perform a sum reduction
req = Iallreduce!(X, MPI.SUM, comm)
sleep(rand())
MPI.Wait(req)

if MPI.Comm_rank(comm) == root
    println("Rank $(MPI.Comm_rank(comm)): The array is: ", X)
    println("Rank $(MPI.Comm_rank(comm)): Should have gotten: ", sum(0:MPI.Comm_size(comm)-1))
end

MPI.Finalize()