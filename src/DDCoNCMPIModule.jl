"""
    DDCoNCMPIModule  

Module for operations on partitions of finite element models for CG solves based
on the Coherent Nodal Clusters.

Implementation for MPI execution.

The module provides the `PartitionedVector` as a mechanism for working with
partitions. The partitioned vector defines a mechanism for storing and
communicating data for each partition. The data is stored in two buffers: one
for the non-shared degrees of, and one for the extended degrees of freedom. The
buffers are stored in a vector of vectors, one for each partition.

The non-shared degrees of freedom are the degrees of freedom that are owned
exclusively by a partition. Nevertheless, the partition needs also access to
degrees of freedom owned by other partitions.

(i) Multiplication of the stiffness matrix by a vector. The non-shared
partitioning is used. 

(ii) Preparation of the two-level preconditioner. The non-shared partitioning is
used. 

(iii) Solution of the systems of equations at the level of the partitions (level
1 of the two-level Schwarz preconditioner). The extended partitioning is used.

Here, the local degrees of freedom numbering is such that
`p.partition_list[i].entity_list.nonshared.ldofs_other[j]` are degrees
of freedom that partition `i` receives from partition `j`, and
`p.partition_list[j].entity_list.nonshared.ldofs_self[i]` are degrees of
freedom that partition `j` sends to partition `i`.

`_lhs_update!` and `_rhs_update!` are functions that update the nonshared
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
using MPI

using ..PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData, npartitions
using ..FinEtoolsDDMethods: set_up_timers, update_timer!, reset_timers!

import ..CGModule: vec_copyto!
import ..CGModule: vec_aypx!
import ..CGModule: vec_ypax!
import ..CGModule: vec_dot
import Base: deepcopy

torank(i) = i - 1
topartitionnumber(i) = i + 1

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
    ns::Vector{T}
    xt::Vector{T}
    recv::Vector{Vector{T}}
    send::Vector{Vector{T}}
end

struct PartitionedVector{DDC<:DDCoNCMPIComm, T<:Number}
    ddcomm::DDC
    buffers::_Buffers{T}
end

function _make_buffers(::Type{T}, partition) where {T}
    ns = fill(zero(T), length(partition.entity_list.nonshared.global_dofs))
    xt = fill(zero(T), length(partition.entity_list.extended.global_dofs))
    elns = partition.entity_list.nonshared
    elxt = partition.entity_list.extended
    lengths = [
        max(length(elns.ldofs_other[j]), length(elns.ldofs_self[j]),
            length(elxt.ldofs_other[j]), length(elxt.ldofs_self[j]))
        for j in eachindex(elns.ldofs_other)
    ]
    recv = [
        fill(zero(T), lengths[j]) for j in eachindex(elns.ldofs_other)
    ]
    send = [
        fill(zero(T), lengths[j]) for j in eachindex(elns.ldofs_self)
    ]
    return _Buffers(ns, xt, recv, send)
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
    a.buffers.ns .= v   
    a
end

"""
    vec_copyto!(a::PV, v::Vector{T}) where {PV<:PartitionedVector, T}

Copy a global vector into a partitioned vector.
"""
function vec_copyto!(a::PV, v::Vector{T}) where {PV<:PartitionedVector, T}
    partition = a.ddcomm.partition
    a.buffers.ns .= v[partition.entity_list.nonshared.global_dofs]
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
        MPI.send(a.buffers.ns, comm; dest=0)
    end
    if rank == 0
        el = loel[1]
        ownd = el.nonshared.global_dofs[1:el.nonshared.num_own_dofs]
        lod = el.nonshared.dof_glob2loc[ownd]
        v[ownd] .= a.buffers.ns[lod]
        for r in 1:Np-1
            ns = MPI.recv(comm; source=r)
            i = r + 1
            el = loel[i]
            ownd = el.nonshared.global_dofs[1:el.nonshared.num_own_dofs]
            lod = el.nonshared.dof_glob2loc[ownd]
            v[ownd] .= ns[lod]
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
    @. y.buffers.ns = x.buffers.ns
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
    @. y.buffers.ns = a * y.buffers.ns + x.buffers.ns
    y
end

"""
    vec_ypax!(y::PV, a, x::PV) where {PV<:PartitionedVector}

Compute `y = y + a x`.
"""
function vec_ypax!(y::PV, a, x::PV) where {PV<:PartitionedVector}
    @. y.buffers.ns = y.buffers.ns + a * x.buffers.ns
    y
end

"""
    vec_dot(x::PV, y::PV) where {PV<:PartitionedVector}

Compute the dot product of two partitioned vectors.
"""
function vec_dot(x::PV, y::PV) where {PV<:PartitionedVector}
    partition = y.ddcomm.partition
    own = 1:partition.entity_list.nonshared.num_own_dofs
    result = dot(y.buffers.ns[own], x.buffers.ns[own])
    return MPI.Allreduce(result, MPI.SUM, y.ddcomm.comm)
end

function _rhs_update!(p::PV) where {PV<:PartitionedVector}
    comm = p.ddcomm.comm
    partition = p.ddcomm.partition
    # Start all sends
    bs = p.buffers
    ldofs_self = partition.entity_list.nonshared.ldofs_self
    ldofs_other = partition.entity_list.nonshared.ldofs_other
    # Start all receives
    requests = MPI.Request[]
    for other in eachindex(ldofs_other)
        if !isempty(ldofs_other[other])
            n = length(ldofs_other[other])
            push!(requests, MPI.Irecv!(view(bs.recv[other], 1:n), comm; source=torank(other)))
        end
    end
    # Start all sends
    for other in eachindex(ldofs_self)
        if !isempty(ldofs_self[other])
            n = length(ldofs_self[other])
            bs.send[other][1:n] .= bs.ns[ldofs_self[other]]
            push!(requests, MPI.Isend(view(bs.send[other], 1:n), comm; dest=torank(other)))
        end
    end
    # Wait for all receives, unpack. 
    MPI.Waitall(requests)
    for other in eachindex(ldofs_other)
        if !isempty(ldofs_other[other])
            n = length(ldofs_other[other])
            bs.ns[ldofs_other[other]] .= bs.recv[other][1:n]
        end
    end
    # println("$(partition.rank): after bs ns = $(bs.ns)")
    # while true
    #     other = MPI.Waitany(requests)
    #     if other === nothing
    #         break
    #     end
    #     n = length(ldofs_other[other])
    #     bs.ns[ldofs_other[other]] .= bs.recv[other][1:n]
    # end
end

function _lhs_update!(q::PV) where {PV<:PartitionedVector}
    comm = q.ddcomm.comm
    partition = q.ddcomm.partition    
    bs = q.buffers
    ldofs_other = partition.entity_list.nonshared.ldofs_other
    ldofs_self = partition.entity_list.nonshared.ldofs_self
    # Start all receives
    requests = MPI.Request[]
    for other in eachindex(ldofs_self)
        if !isempty(ldofs_self[other])
            n = length(ldofs_self[other])
            push!(requests, MPI.Irecv!(view(bs.recv[other], 1:n), comm; source=torank(other)))
        end
    end
    # Start all sends
    for other in eachindex(ldofs_other)
        if !isempty(ldofs_other[other])
            n = length(ldofs_other[other])
            bs.send[other][1:n] .= bs.ns[ldofs_other[other]]
            push!(requests, MPI.Isend(view(bs.send[other], 1:n), comm; dest=torank(other)))
        end
    end
    # Wait for all receives, unpack. 
    MPI.Waitall(requests)
    for other in eachindex(ldofs_self)
        if !isempty(ldofs_self[other])
            n = length(ldofs_self[other])
            bs.ns[ldofs_self[other]] .+= bs.recv[other][1:n]
        end
    end
    # while true
    #     other = MPI.Waitany(requests)
    #     println("$(partition.rank): Got $(torank(other)), $(requests)")
    #     if other === nothing
    #         break
    #     end
    #     n = length(ldofs_self[other])
    #     bs.ns[ldofs_self[other]] .+= bs.recv[other][1:n]
    #     println("$(partition.rank) received from $(torank(other)): $(bs.recv[other][1:n])")
    # end
end

function aop!(q::PV, p::PV) where {PV<:PartitionedVector}
    _rhs_update!(p)
    partition = p.ddcomm.partition
    q.buffers.ns .= partition.Kns_ff * p.buffers.ns
    _lhs_update!(q)  
    q    
end

struct TwoLevelPreConditioner{DDC<:DDCoNCMPIComm, T, IT, FACTOR}
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
    i = rank + 1
    pel = ddcomm.list_of_entity_lists[i].nonshared
    # First we work with all the degrees of freedom on the partition
    P = Phi[pel.global_dofs, :]
    Kr_ff += (P' * partition.Kns_ff * P)
    # Now we need to add all those sparse matrices together. Here it is done on
    # the root process and the result is farmed out to all the other
    # partitions.
    ks = MPI.gather(Kr_ff, comm; root=0)
    if rank == 0
        Kr_ff = spzeros(nr, nr)
        for k in ks
            Kr_ff += k
        end
        for i in 1:MPI.Comm_size(comm)-1
            MPI.send(Kr_ff, comm; dest=i)
        end
    end
    if rank > 0
        Kr_ff = MPI.recv(comm; source=0)
    end
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
    # Narrow by the transformation 
    ld = partition.entity_list.nonshared.ldofs_own_only
    pre.buffPp .= pre.buff_Phi' * p.buffers.ns[ld]
    # Communicate
    pre.buffPp .= MPI.Allreduce!(pre.buffPp, MPI.SUM, pre.ddcomm.comm)
    # Solve the reduced problem
    pre.buffKiPp .= pre.Kr_ff_factor \ pre.buffPp
    # Expand by the transformation 
    q.buffers.ns .= 0
    ld = partition.entity_list.nonshared.ldofs_own_only
    q.buffers.ns[ld] .= pre.buff_Phi * pre.buffKiPp
    _lhs_update!(q)
    q.buffers.xt .= partition.Kxt_ff_factor \ p.buffers.xt
    _lhs_update_xt!(q)
    q
end

function _rhs_update_xt!(p::PV) where {PV<:PartitionedVector}
    comm = p.ddcomm.comm
    partition = p.ddcomm.partition
    bs = p.buffers
    ldofs_self = partition.entity_list.extended.ldofs_self
    ldofs_other = partition.entity_list.extended.ldofs_other
    # Copy data from nonshared to extended
    psx = partition.entity_list.extended
    bs.xt .= 0
    ld = psx.ldofs_own_only
    bs.xt[ld] .= bs.ns[ld]
    # Start all receives
    requests = MPI.Request[]
    for other in eachindex(ldofs_other)
        if !isempty(ldofs_other[other])
            n = length(ldofs_other[other])
            push!(requests, MPI.Irecv!(view(bs.recv[other], 1:n), comm; source=torank(other)))
        end
    end
    # Start all sends
    for other in eachindex(ldofs_self)
        if !isempty(ldofs_self[other])
            n = length(ldofs_self[other])
            bs.send[other][1:n] .= bs.xt[ldofs_self[other]]
            push!(requests, MPI.Isend(view(bs.send[other], 1:n), comm; dest=torank(other)))
        end
    end
    # Wait for all receives, unpack. 
    MPI.Waitall(requests)
    for other in eachindex(ldofs_other)
        if !isempty(ldofs_other[other])
            n = length(ldofs_other[other])
            bs.xt[ldofs_other[other]] .= bs.recv[other][1:n]
        end
    end
    # while true
    #     other = MPI.Waitany(requests)
    #     if other === nothing
    #         break
    #     end
    #     n = length(ldofs_other[other])
    #     bs.xt[ldofs_other[other]] .= bs.recv[other][1:n]
    # end
end

function _lhs_update_xt!(q::PV) where {PV<:PartitionedVector}
    comm = q.ddcomm.comm
    partition = q.ddcomm.partition
    bs = q.buffers
    ldofs_other = partition.entity_list.extended.ldofs_other
    ldofs_self = partition.entity_list.extended.ldofs_self
    # Start all receives
    requests = MPI.Request[]
    for other in eachindex(ldofs_self)
        if !isempty(ldofs_self[other])
            n = length(ldofs_self[other])
            push!(requests, MPI.Irecv!(view(bs.recv[other], 1:n), comm; source=torank(other)))
        end
    end
    # Start all sends
    for other in eachindex(ldofs_other)
        if !isempty(ldofs_other[other])
            n = length(ldofs_other[other])
            bs.send[other][1:n] .= bs.xt[ldofs_other[other]]
            push!(requests, MPI.Isend(view(bs.send[other], 1:n), comm; dest=torank(other)))
        end
    end
    # Wait for all receives, unpack. 
    MPI.Waitall(requests)
    for other in eachindex(ldofs_self)
        if !isempty(ldofs_self[other])
            n = length(ldofs_self[other])
            bs.xt[ldofs_self[other]] .+= bs.recv[other][1:n]
        end
    end
    # while true
    #     other = MPI.Waitany(requests)
    #     if other === nothing
    #         break
    #     end
    #     n = length(ldofs_other[other])
    #     bs.xt[ldofs_other[other]] .= bs.recv[other][1:n]
    # end
    ld = partition.entity_list.nonshared.ldofs_own_only
    bs.ns[ld] .+= bs.xt[ld]
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
