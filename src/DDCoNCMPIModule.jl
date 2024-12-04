"""
    DDCoNCMPIModule  

Module for operations on partitions of finite element models for CG solves based
on the Coherent Nodal Clusters.

Implementation for MPI execution.

The module provides the `PartitionedVector` as a mechanism for working with
partitions. The partitioned vector defines a mechanism for storing and
communicating data for each partition. The data is stored in two buffers: one
for the owned degrees of, and one for the extended degrees of freedom. The
buffers are stored in a vector of vectors, one for each partition.

The owned degrees of freedom are the degrees of freedom that are owned
exclusively by a partition. Nevertheless, the partition needs also access to
degrees of freedom owned by other partitions.

(i) Multiplication of the stiffness matrix by a vector. The owned
partitioning is used. 

(ii) Preparation of the two-level preconditioner. The owned partitioning is
used. 

(iii) Solution of the systems of equations at the level of the partitions (level
1 of the two-level Schwarz preconditioner). The extended partitioning is used.

Here, the local degrees of freedom numbering is such that
`p.partition_list[i].entity_list.own.ldofs_other[j]` are degrees
of freedom that partition `i` receives from partition `j`, and
`p.partition_list[j].entity_list.own.ldofs_self[i]` are degrees of
freedom that partition `j` sends to partition `i`.

`_lhs_update!` and `_rhs_update!` are functions that update the owned
degrees of freedom. The first is analogous to scatter, whereas the second is
analogous to gather. 

Scatter: owned degrees of freedom are known but the partition
needs to receive the values of the degrees of freedom that it doesn't own.

Gather: the partition needs to update the values of the degrees of freedom that
it owns, receiving contributions to those degrees of freedom from other
partitions.
"""
module DDCoNCMPIModule

__precompile__(true)

using FinEtools
using CoNCMOR
using SparseArrays
using Krylov
using Metis
using LinearOperators
using SparseArrays
using LinearAlgebra
using Statistics: mean
using ..FENodeToPartitionMapModule: FENodeToPartitionMap
using ShellStructureTopo
using ..PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData, npartitions
using ..FinEtoolsDDMethods: set_up_timers, update_timer!, reset_timers!
import ..CGModule: vec_copyto!
import ..CGModule: vec_aypx!
import ..CGModule: vec_ypax!
import ..CGModule: vec_dot
import Base: deepcopy

using MPI

torank(i) = i - 1
topartitionnumber(r) = r + 1

"""
    DDCoNCMPIComm{MPIC, EL, PD<:CoNCPartitionData}

Communicator for MPI execution.
"""
struct DDCoNCMPIComm{MPIC, EL, PD<:CoNCPartitionData}
    comm::MPIC
    list_of_entity_lists::EL
    partition::PD
    gdim::Int
end

function DDCoNCMPIComm(comm, cpi, fes, make_matrix, make_interior_load)
    rank = MPI.Comm_rank(comm)
    partition = CoNCPartitionData(cpi, rank, fes, make_matrix, make_interior_load)
    gdim = nfreedofs(cpi.u)
    return DDCoNCMPIComm(comm, cpi.list_of_entity_lists, partition, gdim)
end

struct _Buffers{T}
    ownv::Vector{T}
    extv::Vector{T}
    vrecvv::Vector{Vector{T}}
    vsendv::Vector{Vector{T}}
end

struct PartitionedVector{DDC<:DDCoNCMPIComm, T<:Number}
    ddcomm::DDC
    buffers::_Buffers{T}
end

function _make_buffers(::Type{T}, partition) where {T}
    ownv = fill(zero(T), length(partition.entity_list.own.global_dofs))
    extv = fill(zero(T), length(partition.entity_list.extended.global_dofs))
    ownel = partition.entity_list.own
    extel = partition.entity_list.extended
    lengths = [
        max(length(ownel.ldofs_other[j]), length(ownel.ldofs_self[j]),
            length(extel.ldofs_other[j]), length(extel.ldofs_self[j]))
        for j in eachindex(ownel.ldofs_other)
    ]
    vrecvv = [
        fill(zero(T), lengths[j]) for j in eachindex(ownel.ldofs_other)
    ]
    vsendv = [
        fill(zero(T), lengths[j]) for j in eachindex(ownel.ldofs_self)
    ]
    return _Buffers(ownv, extv, vrecvv, vsendv)
end

function PartitionedVector(::Type{T}, ddcomm::DDC) where {T, DDC<:DDCoNCMPIComm}
    partition = ddcomm.partition
    buffers = _make_buffers(T, partition)
    return PartitionedVector(ddcomm, buffers) 
end

"""
    vec_copyto!(a::PV, v::T) where {PV<:PartitionedVector, T}

Copy a single value into the partitioned vector.
"""
function vec_copyto!(a::PV, v::T) where {PV<:PartitionedVector, T}
    a.buffers.ownv .= v   
    a
end

"""
    vec_copyto!(a::PV, v::Vector{T}) where {PV<:PartitionedVector, T}

Copy a global vector into a partitioned vector.
"""
function vec_copyto!(a::PV, v::Vector{T}) where {PV<:PartitionedVector, T}
    partition = a.ddcomm.partition
    a.buffers.ownv .= v[partition.entity_list.own.global_dofs]
    a
end

"""
    vec_copyto!(v::Vector{T}, a::PV) where {PV<:PartitionedVector, T}

Copy a partitioned vector into a global vector.
"""
function vec_copyto!(v::Vector{T}, a::PV) where {PV<:PartitionedVector, T}
    error("Not implemented")
    v
end

"""
    vec_collect(a::PV) where {PV<:PartitionedVector}

Collect a partitioned vector into a global vector.
"""
function vec_collect(inp::PV) where {PV<:PartitionedVector}
    a = deepcopy(inp)
    comm = a.ddcomm.comm
    Np = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    loel = a.ddcomm.list_of_entity_lists
    gdim = a.ddcomm.gdim
    v = zeros(gdim)
    if rank > 0
        MPI.send(a.buffers.ownv, comm; dest=0)
    end
    if rank == 0
        el = loel[1]
        ownd = el.own.global_dofs[1:el.own.num_own_dofs]
        lod = el.own.dof_glob2loc[ownd]
        v[ownd] .= a.buffers.ownv[lod]
        for r in 1:Np-1
            ov = MPI.recv(comm; source=r)
            i = r + 1
            el = loel[i]
            ownd = el.own.global_dofs[1:el.own.num_own_dofs]
            lod = el.own.dof_glob2loc[ownd]
            v[ownd] .= ov[lod]
        end
    end
    return MPI.bcast(v, comm; root=0)
end

"""
    vec_copyto!(y::PV, x::PV) where {PV<:PartitionedVector}

Copy one partitioned vector to another.

The contents of `x` is copied into `y`.
"""
function vec_copyto!(y::PV, x::PV) where {PV<:PartitionedVector}
    @. y.buffers.ownv = x.buffers.ownv
    y
end

"""
    deepcopy(a::PV) where {PV<:PartitionedVector}

Create a deep copy of a partitioned vector.
"""
function deepcopy(a::PV) where {PV<:PartitionedVector}
    return PartitionedVector(a.ddcomm, deepcopy(a.buffers))
end

"""
    vec_aypx!(y::PV, a, x::PV) where {PV<:PartitionedVector}

Compute `y = a y + x`.
"""
function vec_aypx!(y::PV, a, x::PV) where {PV<:PartitionedVector}
    @. y.buffers.ownv = a * y.buffers.ownv + x.buffers.ownv
    y
end

"""
    vec_ypax!(y::PV, a, x::PV) where {PV<:PartitionedVector}

Compute `y = y + a x`.
"""
function vec_ypax!(y::PV, a, x::PV) where {PV<:PartitionedVector}
    @. y.buffers.ownv = y.buffers.ownv + a * x.buffers.ownv
    y
end

"""
    vec_dot(x::PV, y::PV) where {PV<:PartitionedVector}

Compute the dot product of two partitioned vectors.
"""
function vec_dot(x::PV, y::PV) where {PV<:PartitionedVector}
    partition = y.ddcomm.partition
    own = 1:partition.entity_list.own.num_own_dofs
    result = dot(y.buffers.ownv[own], x.buffers.ownv[own])
    return MPI.Allreduce(result, MPI.SUM, y.ddcomm.comm)
end

function _rhs_update!(p::PV) where {PV<:PartitionedVector}
    comm = p.ddcomm.comm
    partition = p.ddcomm.partition
    # Start all sends
    bs = p.buffers
    ldofs_self = partition.entity_list.own.ldofs_self
    ldofs_other = partition.entity_list.own.ldofs_other
    # Start all receives
    recvrequests = MPI.Request[]
    for other in eachindex(ldofs_other)
        if !isempty(ldofs_other[other])
            n = length(ldofs_other[other])
            push!(recvrequests, MPI.Irecv!(view(bs.vrecvv[other], 1:n), comm; source=torank(other)))
        end
    end
    # Start all sends
    sendrequests = MPI.Request[]
    for other in eachindex(ldofs_self)
        if !isempty(ldofs_self[other])
            n = length(ldofs_self[other])
            bs.vsendv[other][1:n] .= bs.ownv[ldofs_self[other]]
            push!(sendrequests, MPI.Isend(view(bs.vsendv[other], 1:n), comm; dest=torank(other)))
        end
    end
    # Wait for all receives, unpack. 
    MPI.Waitall(sendrequests)
    status = Ref(MPI.STATUS_ZERO)
    while true
        idx = MPI.Waitany(recvrequests, status)
        if idx === nothing
            break
        end
        other = topartitionnumber(status[].source)
        n = length(ldofs_other[other])
        bs.ownv[ldofs_other[other]] .= bs.vrecvv[other][1:n]
    end
end

function _lhs_update!(q::PV) where {PV<:PartitionedVector}
    comm = q.ddcomm.comm
    partition = q.ddcomm.partition    
    bs = q.buffers
    ldofs_other = partition.entity_list.own.ldofs_other
    ldofs_self = partition.entity_list.own.ldofs_self
    # Start all receives
    recvrequests = MPI.Request[]
    for other in eachindex(ldofs_self)
        if !isempty(ldofs_self[other])
            n = length(ldofs_self[other])
            push!(recvrequests, MPI.Irecv!(view(bs.vrecvv[other], 1:n), comm; source=torank(other)))
        end
    end
    # Start all sends
    sendrequests = MPI.Request[]
    for other in eachindex(ldofs_other)
        if !isempty(ldofs_other[other])
            n = length(ldofs_other[other])
            bs.vsendv[other][1:n] .= bs.ownv[ldofs_other[other]]
            push!(sendrequests, MPI.Isend(view(bs.vsendv[other], 1:n), comm; dest=torank(other)))
        end
    end
    MPI.Waitall(sendrequests)
    # Wait for all receives, unpack. 
    status = Ref(MPI.STATUS_ZERO)
    while true
        idx = MPI.Waitany(recvrequests, status)
        if idx === nothing
            break
        end
        other = topartitionnumber(status[].source)
        n = length(ldofs_self[other])
        bs.ownv[ldofs_self[other]] .+= bs.vrecvv[other][1:n]
    end
end

function aop!(q::PV, p::PV) where {PV<:PartitionedVector}
    _rhs_update!(p)
    partition = p.ddcomm.partition
    q.buffers.ownv .= partition.Kown_ff * p.buffers.ownv
    _lhs_update!(q)  
    q    
end

mutable struct TwoLevelPreConditioner{DDC<:DDCoNCMPIComm, T, IT, FACTOR}
    ddcomm::DDC
    n::IT
    buff_Phi::SparseMatrixCSC{T, IT}
    Kr_ff_factor::FACTOR
    buffPp::Vector{T}
    buffKiPp::Vector{T}
end

function TwoLevelPreConditioner(ddcomm::DDC, Phi) where {DDC<:DDCoNCMPIComm}
    comm = ddcomm.comm
    partition = ddcomm.partition
    rank = ddcomm.partition.rank 
    n = size(Phi, 1)
    nr = size(Phi, 2)
    Kr_ff = spzeros(nr, nr)
    i = topartitionnumber(rank)
    pel = ddcomm.list_of_entity_lists[i].own
    # First we work with all the degrees of freedom on the partition
    P = Phi[pel.global_dofs, :]
    Kr_ff += (P' * partition.Kown_ff * P)
    # Now we need to add all those sparse matrices together. Here it is done on
    # the root process and the result is farmed out to all the other
    # partitions.
    ks = MPI.gather(Kr_ff, comm; root=0)
    if rank == 0
        Kr_ff = spzeros(nr, nr)
        for k in ks
            Kr_ff += k
        end
    end
    Kr_ff = MPI.bcast(Kr_ff, 0, comm)
    Kr_ff_factor = lu(Kr_ff)
    # the transformation matrices are now resized to only the own dofs
    buff_Phi = P[pel.ldofs_own_only, :]
    buffPp = fill(zero(eltype(Kr_ff_factor)), nr)
    buffKiPp = fill(zero(eltype(Kr_ff_factor)), nr)
    return TwoLevelPreConditioner(ddcomm, n, buff_Phi, Kr_ff_factor, buffPp, buffKiPp)
end

function (pre::TwoLevelPreConditioner)(q::PV, p::PV) where {PV<:PartitionedVector}
    partition = p.ddcomm.partition
    _rhs_update_xt!(p)
    q.buffers.ownv .= 0
    # Level 2, narrow by the transformation 
    ld = partition.entity_list.own.ldofs_own_only
    pre.buffPp .= pre.buff_Phi' * p.buffers.ownv[ld]
    # Level 2, communicate
    req = MPI.Iallreduce!(pre.buffPp, MPI.SUM, pre.ddcomm.comm)
    # Level 1
    q.buffers.extv .= partition.Kxt_ff_factor \ p.buffers.extv
    # Level 2, wait for the communication
    MPI.Wait(req)
    # Level 2, solve the reduced problem
    pre.buffKiPp .= pre.Kr_ff_factor \ pre.buffPp
    # Level 2, expand by the transformation 
    ld = partition.entity_list.own.ldofs_own_only
    q.buffers.ownv[ld] .= pre.buff_Phi * pre.buffKiPp
    
    _lhs_update_xt!(q)
    q
end

function _rhs_update_xt!(p::PV) where {PV<:PartitionedVector}
    comm = p.ddcomm.comm
    partition = p.ddcomm.partition
    bs = p.buffers
    ldofs_self = partition.entity_list.extended.ldofs_self
    ldofs_other = partition.entity_list.extended.ldofs_other
    # Copy data from owned to extended
    psx = partition.entity_list.extended
    bs.extv .= 0
    ld = psx.ldofs_own_only
    bs.extv[ld] .= bs.ownv[ld]
    # Start all receives
    recvrequests = MPI.Request[]
    for other in eachindex(ldofs_other)
        if !isempty(ldofs_other[other])
            n = length(ldofs_other[other])
            push!(recvrequests, MPI.Irecv!(view(bs.vrecvv[other], 1:n), comm; source=torank(other)))
        end
    end
    # Start all sends
    sendrequests = MPI.Request[]
    for other in eachindex(ldofs_self)
        if !isempty(ldofs_self[other])
            n = length(ldofs_self[other])
            bs.vsendv[other][1:n] .= bs.extv[ldofs_self[other]]
            push!(sendrequests, MPI.Isend(view(bs.vsendv[other], 1:n), comm; dest=torank(other)))
        end
    end
    MPI.Waitall(sendrequests)
    # Wait for all receives, unpack. 
    status = Ref(MPI.STATUS_ZERO)
    while true
        idx = MPI.Waitany(recvrequests, status)
        if idx === nothing
            break
        end
        other = topartitionnumber(status[].source)
        n = length(ldofs_other[other])
        bs.extv[ldofs_other[other]] .= bs.vrecvv[other][1:n]
    end
end

function _lhs_update_xt!(q::PV) where {PV<:PartitionedVector}
    comm = q.ddcomm.comm
    partition = q.ddcomm.partition
    bs = q.buffers
    ldofs_other = partition.entity_list.extended.ldofs_other
    ldofs_self = partition.entity_list.extended.ldofs_self
    # Start all receives
    recvrequests = MPI.Request[]
    for other in eachindex(ldofs_self)
        if !isempty(ldofs_self[other])
            n = length(ldofs_self[other])
            push!(recvrequests, MPI.Irecv!(view(bs.vrecvv[other], 1:n), comm; source=torank(other)))
        end
    end
    # Start all sends
    sendrequests = MPI.Request[]
    for other in eachindex(ldofs_other)
        if !isempty(ldofs_other[other])
            n = length(ldofs_other[other])
            bs.vsendv[other][1:n] .= bs.extv[ldofs_other[other]]
            push!(sendrequests, MPI.Isend(view(bs.vsendv[other], 1:n), comm; dest=torank(other)))
        end
    end
    MPI.Waitall(sendrequests)
    # Wait for all receives, unpack. 
    status = Ref(MPI.STATUS_ZERO)
    while true
        idx = MPI.Waitany(recvrequests, status)
        if idx === nothing
            break
        end
        other = topartitionnumber(status[].source)
        n = length(ldofs_self[other])
        bs.extv[ldofs_self[other]] .+= bs.vrecvv[other][1:n]
    end
    # Update the owned
    ld = partition.entity_list.own.ldofs_own_only
    bs.ownv[ld] .+= bs.extv[ld]
end

function rhs(ddcomm::DDC) where {DDC<:DDCoNCMPIComm} 
    comm = ddcomm.comm
    partition = ddcomm.partition
    rank = partition.rank
    rhss = MPI.gather(partition.rhs, comm; root=0)
    if rank == 0
        rhs = zeros(eltype(rhss[1]), length(rhss[1]))
        for r in rhss
            rhs += r
        end
        for i in 1:MPI.Comm_size(comm)-1
            MPI.send(rhs, comm; dest=i)
        end
    end
    if rank > 0
        rhs = MPI.recv(comm; source=0)
    end
    return rhs
end

end # module DDCoNCMPIModule
