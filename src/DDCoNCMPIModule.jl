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

torank(i) = i - 1

"""
    DDCoNCMPIComm{MPIC, EL, PD<:CoNCPartitionData}

Communicator for MPI execution.
"""
struct DDCoNCMPIComm{MPIC, EL, PD<:CoNCPartitionData}
    comm::MPIC
    list_of_entity_lists::EL
    partition::PD
end

function DDCoNCMPIComm(comm, cpi, fes, make_matrix, make_interior_load)
    rank = MPI.Comm_rank(comm)
    partition = CoNCPartitionData(cpi, rank, fes, make_matrix, make_interior_load)
    return DDCoNCMPIComm(comm, cpi.list_of_entity_lists, partition)
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
    error("Implementation under review")
    partition = a.ddcomm.partition
    el = partition.entity_list
    ownd = el.nonshared.global_dofs[1:el.nonshared.num_own_dofs]
    lod = el.nonshared.dof_glob2loc[ownd]
    v[ownd] .= a.buffers.ns[lod]
    v
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

# Computes y = x + a y.
function vec_aypx!(y::PV, a, x::PV) where {PV<:PartitionedVector}
    @. y.buffers.ns = a * y.buffers.ns + x.buffers.ns
    y
end

# Computes y += a x
function vec_ypax!(y::PV, a, x::PV) where {PV<:PartitionedVector}
    @. y.buffers.ns = y.buffers.ns + a * x.buffers.ns
    y
end

function vec_dot(x::PV, y::PV) where {PV<:PartitionedVector}
    partition = y.ddcomm.partition
    own = 1:partition.entity_list.nonshared.num_own_dofs
    result = dot(y.buffers.ns[own], x.buffers.ns[own])
    return MPI.Allreduce(result, MPI.SUM, y.ddcomm.comm)
end

function _rhs_update!(p::PV) where {PV<:PartitionedVector}
    comm = p.ddcomm.comm
    loel = p.ddcomm.list_of_entity_lists
    i = p.ddcomm.partition.rank + 1
    # First I will send to other partitions
    ldofs_self = loel[i].nonshared.ldofs_self
    for j in eachindex(ldofs_self)
        if !isempty(ldofs_self[j])
            req = MPI.Isend(p.buff_ns[ldofs_self[j]], comm; dest = torank(j))
        end
    end
    # Next I will receive from other partitions
    requests = MPI.Request[]
    ldofs_other = loel[i].nonshared.ldofs_other
    for j in eachindex(ldofs_other)
        if !isempty(ldofs_other[j])
            push!(requests, MPI.Irecv!(p.buff_ns[ldofs_other[j]], comm; source = torank(j)))
        end
    end
    MPI.Waitall(requests)
end

function _lhs_update!(q::PV) where {PV<:PartitionedVector}
    comm = q.ddcomm.comm
    loel = q.ddcomm.list_of_entity_lists
    i = q.ddcomm.partition.rank + 1
    # First I will send to other partitions
    ldofs_other = loel[i].nonshared.ldofs_other
    for j in eachindex(ldofs_other)
        if !isempty(ldofs_other[j])
            req = MPI.Isend(q.buff_ns[ldofs_other[j]], comm; dest = torank(j))
        end
    end
    # Next I will receive from other partitions
    requests = MPI.Request[]
    ldofs_self = loel[i].nonshared.ldofs_self
    for j in eachindex(ldofs_self)
        if !isempty(ldofs_self[j])
            q.temp_ns[j] .= 0
            push!(requests, MPI.Irecv!(q.temp_ns[j][ldofs_self[j]], comm; source = torank(j)))
        end
    end
    MPI.Waitall(requests)
    q.buff_ns .= 0
    for j in eachindex(ldofs_self)
        if !isempty(ldofs_self[j])
            @. q.buff_ns += q.temp_ns[j]
        end
    end
end

function aop!(q::PV, p::PV) where {PV<:PartitionedVector}
    _rhs_update!(p)
    partition = p.ddcomm.partition
    q.buff_ns .= partition.Kns_ff * p.buff_ns
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
    comm = ddcomm.comm
    partition = ddcomm.partition
    rank = ddcomm.partition.rank 
    rhs_update_xt!(p)
    pre.buffPp .= zero(eltype(pre.buffPp))
    for i in eachindex(partition_list)
        ld = partition_list[i].entity_list.nonshared.ldofs_own_only
        pre.buffPp .+= pre.buff_Phis[i]' * p.buff_ns[i][ld]
    end
    pre.buffKiPp .= pre.Kr_ff_factor \ pre.buffPp
    for i in eachindex(partition_list)
        q.buff_ns[i] .= 0
        ld = partition_list[i].entity_list.nonshared.ldofs_own_only
        q.buff_ns[i][ld] .= pre.buff_Phis[i] * pre.buffKiPp
    end
    _lhs_update!(q)
    for i in eachindex(partition_list)
        q.buff_xt[i] .= partition_list[i].Kxt_ff_factor \ p.buff_xt[i]
    end
    lhs_update_xt!(q)
    q
end

function rhs_update_xt!(p::PV) where {PV<:PartitionedVector}
    partition = p.ddcomm.partition
    pie = partition.entity_list.extended
    ld = pie.ldofs_own_only
    p.buff_xt .= 0
    p.buff_xt[ld] .= p.buff_ns[ld]
    # First I will send to other partitions
    ldofs_self = loel[i].extended.ldofs_self
    for j in eachindex(ldofs_self)
        if !isempty(ldofs_self[j])
            req = MPI.Isend(p.buff_xt[ldofs_self[j]], comm; dest = torank(j))
        end
    end
    # Next I will receive from other partitions
    requests = MPI.Request[]
    ldofs_other = loel[i].extended.ldofs_other
    for j in eachindex(ldofs_other)
        if !isempty(ldofs_other[j])
            push!(requests, MPI.Irecv!(p.buff_xt[ldofs_other[j]], comm; source = torank(j)))
        end
    end
    MPI.Waitall(requests)
end

function lhs_update_xt!(q::PV) where {PV<:PartitionedVector}
    comm = q.ddcomm.comm
    loel = q.ddcomm.list_of_entity_lists
    i = q.ddcomm.partition.rank + 1
    # First I will send to other partitions
    ldofs_other = loel[i].extended.ldofs_other
    for j in eachindex(ldofs_other)
        if !isempty(ldofs_other[j])
            req = MPI.Isend(q.buff_xt[ldofs_other[j]], comm; dest = torank(j))
        end
    end
    # Next I will receive from other partitions
    requests = MPI.Request[]
    ldofs_self = loel[i].nonshared.ldofs_self
    for j in eachindex(ldofs_self)
        if !isempty(ldofs_self[j])
            q.temp_ns[j] .= 0
            push!(requests, MPI.Irecv!(q.temp_ns[j][ldofs_self[j]], comm; source = torank(j)))
        end
    end
    MPI.Waitall(requests)
    q.buff_ns .= 0
    for j in eachindex(ldofs_self)
        if !isempty(ldofs_self[j])
            @. q.buff_ns += q.temp_ns[j]
        end
    end
    #=
    partition = q.ddcomm.partition
    for i in eachindex(partition_list)
        ldofs_self = partition_list[i].entity_list.extended.ldofs_self
        for j in eachindex(ldofs_self)
            if !isempty(ldofs_self[j])
                ldofs_other = partition_list[j].entity_list.extended.ldofs_other
                q.buff_xt[i][ldofs_self[j]] .+= q.buff_xt[j][ldofs_other[i]]
            end
        end
        qin = partition_list[i].entity_list.nonshared
        ld = qin.ldofs_own_only
        q.buff_ns[i][ld] .+= q.buff_xt[i][ld]
    end
    =#
end

function rhs(comm::C) where {C<:DDCoNCMPIComm} 
    rhs = deepcopy(comm.partition_list[1].rhs)
    rhs .= 0
    for i in eachindex(comm.partition_list)
        rhs .+= comm.partition_list[i].rhs
    end  
    return rhs
end

end # module DDCoNCMPIModule
