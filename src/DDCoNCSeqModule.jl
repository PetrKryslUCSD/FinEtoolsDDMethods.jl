"""
    DDCoNCSeqModule  

Module for operations on partitions of finite element models for CG solves based
on the Coherent Nodal Clusters.

Implementation for sequential execution.

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
`partition_list[i].nonshared.ldofs_other[j]` are degrees
of freedom that partition `i` receives from partition `j`, and
`partition_list[j].nonshared.ldofs_self[i]` are degrees of
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
module DDCoNCSeqModule

__precompile__(true)

using FinEtools
using CoNCMOR
using SparseArrays
using Krylov
using Metis
using LinearOperators
using SparseArrays
using LinearAlgebra
import Base: size, eltype, deepcopy
import LinearAlgebra: mul!
using Statistics: mean
using ..FENodeToPartitionMapModule: FENodeToPartitionMap
using ShellStructureTopo

using ..PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData, npartitions
using ..FinEtoolsDDMethods: set_up_timers, update_timer!, reset_timers!

import ..CGModule: vec_copyto!
import ..CGModule: vec_aypx!
import ..CGModule: vec_ypax!
import ..CGModule: vec_dot

"""
    DDCoNCSeqComm{PD<:CoNCPartitionData}

Communicator for sequential execution.
"""
struct DDCoNCSeqComm{PD<:CoNCPartitionData}
    partition_list::Vector{PD}
end

function DDCoNCSeqComm(comm, cpi, fes, make_matrix, make_interior_load)
    partition_list = _make_partitions(cpi, fes, make_matrix, make_interior_load)
    return DDCoNCSeqComm(partition_list)
end

function _make_partitions(cpi, fes, make_matrix, make_interior_load)
    partition_list  = [CoNCPartitionData(cpi, rank) for rank in 0:1:npartitions(cpi)-1]
    for i in eachindex(partition_list)
        rank = i - 1
        partition_list[i] = CoNCPartitionData(cpi, rank, fes, make_matrix, make_interior_load)
    end  
    return partition_list
end

struct _Buffers{T}
    ns::Vector{T}
    xt::Vector{T}
    onother::Vector{Vector{T}}
    onself::Vector{Vector{T}}
end

struct PartitionedVector{DDC<:DDCoNCSeqComm{PD} where {PD<:CoNCPartitionData}, T<:Number}
    ddcomm::DDC
    buffers::Vector{_Buffers{T}}
end

function _make_buffers(::Type{T}, partition_list) where {T}
    buffers = _Buffers{T}[]
    for partition in partition_list
        ns = fill(zero(T), length(partition.entity_list.nonshared.global_dofs))
        xt = fill(zero(T), length(partition.entity_list.extended.global_dofs))
        elns = partition.entity_list.nonshared
        onother = [
            fill(zero(T), length(elns.ldofs_other[j])) for j in eachindex(elns.ldofs_other)
        ]
        onself = [
            fill(zero(T), length(elns.ldofs_self[j])) for j in eachindex(elns.ldofs_self)
        ]
        push!(buffers, _Buffers(ns, xt, onother, onself))
    end
    return buffers
end

function PartitionedVector(::Type{T}, ddcomm::DDC) where {T, DDC<:DDCoNCSeqComm{PD} where {PD<:CoNCPartitionData}}
    buffers = _make_buffers(T, ddcomm.partition_list)
    return PartitionedVector(ddcomm, buffers)
end

"""
    vec_copyto!(a::PV, v::T) where {PV<:PartitionedVector, T}

Copy a single value into the partitioned vector.
"""
function vec_copyto!(a::PV, v::T) where {PV<:PartitionedVector, T}
    partition_list = a.ddcomm.partition_list
    for i in eachindex(partition_list)
        a.buffers[i].ns .= v   
    end
    a
end

"""
    vec_copyto!(a::PV, v::Vector{T}) where {PV<:PartitionedVector, T}

Copy a global vector into a partitioned vector.
"""
function vec_copyto!(a::PV, v::Vector{T}) where {PV<:PartitionedVector, T}
    partition_list = a.ddcomm.partition_list
    for i in eachindex(partition_list)
        a.buffers[i].ns .= v[partition_list[i].entity_list.nonshared.global_dofs]
    end
    a
end

"""
    vec_copyto!(v::Vector{T}, a::PV) where {PV<:PartitionedVector, T}

Copy a partitioned vector into a global vector.
"""
function vec_copyto!(v::Vector{T}, a::PV) where {PV<:PartitionedVector, T}
    partition_list = a.ddcomm.partition_list
    for i in eachindex(partition_list)
        el = partition_list[i].entity_list
        ownd = el.nonshared.global_dofs[1:el.nonshared.num_own_dofs]
        lod = el.nonshared.dof_glob2loc[ownd]
        v[ownd] .= a.buffers[i].ns[lod]
    end
    v
end

"""
    vec_copyto!(y::PV, x::PV) where {PV<:PartitionedVector}

Copy one partitioned vector to another.

The contents of `x` is copied into `y`.
"""
function vec_copyto!(y::PV, x::PV) where {PV<:PartitionedVector}
    partition_list = y.ddcomm.partition_list
    for i in eachindex(partition_list)
        @. y.buffers[i].ns = x.buffers[i].ns
    end
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
    partition_list = y.ddcomm.partition_list
    for i in eachindex(partition_list)
        @. y.buffers[i].ns = a * y.buffers[i].ns + x.buffers[i].ns
    end
    y
end

# Computes y += a x
function vec_ypax!(y::PV, a, x::PV) where {PV<:PartitionedVector}
    partition_list = y.ddcomm.partition_list
    for i in eachindex(partition_list)
        @. y.buffers[i].ns = y.buffers[i].ns + a * x.buffers[i].ns
    end
    y
end

function vec_dot(x::PV, y::PV) where {PV<:PartitionedVector}
    partition_list = y.ddcomm.partition_list
    result = zero(eltype(x.buffers[1].ns))
    for i in eachindex(partition_list)
        own = 1:partition_list[i].entity_list.nonshared.num_own_dofs
        result += dot(y.buffers[i].ns[own], x.buffers[i].ns[own])
    end
    return result
end

function _rhs_update!(p::PV) where {PV<:PartitionedVector}
    partition_list = p.ddcomm.partition_list
    for self in eachindex(partition_list)
        ldofs_other = partition_list[self].entity_list.nonshared.ldofs_other
        for other in eachindex(ldofs_other)
            if !isempty(ldofs_other[other])
                ldofs_self = partition_list[other].entity_list.nonshared.ldofs_self
                p.buffers[self].ns[ldofs_other[other]] .= p.buffers[other].ns[ldofs_self[self]]
            end
        end
    end
end

function _lhs_update!(q::PV) where {PV<:PartitionedVector}
    partition_list = q.ddcomm.partition_list
    for i in eachindex(partition_list)
        ldofs_self = partition_list[i].entity_list.nonshared.ldofs_self
        for j in eachindex(ldofs_self)
            if !isempty(ldofs_self[j])
                ldofs_other = partition_list[j].entity_list.nonshared.ldofs_other
                q.buffers[i].ns[ldofs_self[j]] .+= q.buffers[j].ns[ldofs_other[i]]
            end
        end
    end
end

function aop!(q::PV, p::PV) where {PV<:PartitionedVector}
    partition_list = p.ddcomm.partition_list
    _rhs_update!(p)
    for i in eachindex(partition_list)
        q.buffers[i].ns .= partition_list[i].Kns_ff * p.buffers[i].ns
    end
    _lhs_update!(q)
    q    
end

struct TwoLevelPreConditioner{DDC<:DDCoNCSeqComm{PD} where {PD<:CoNCPartitionData}, T, IT, FACTOR}
    ddcomm::DDC
    n::IT
    buff_Phis::Vector{SparseMatrixCSC{T, IT}}
    Kr_ff_factor::FACTOR
    buffPp::Vector{T}
    buffKiPp::Vector{T}
end

function TwoLevelPreConditioner(ddcomm::DDC, Phi) where {DDC<:DDCoNCSeqComm{PD} where {PD<:CoNCPartitionData}}
    partition_list = ddcomm.partition_list
    n = size(Phi, 1)
    nr = size(Phi, 2)
    buff_Phis = typeof(Phi)[]
    sizehint!(buff_Phis, length(partition_list))
    Kr_ff = spzeros(nr, nr)
    for i in eachindex(partition_list)
        pel = partition_list[i].entity_list.nonshared
        # First we work with all the degrees of freedom on the partition
        P = Phi[pel.global_dofs, :]
        Kr_ff += (P' * partition_list[i].Kns_ff * P)
        # the transformation matrices are now resized to only the own dofs
        ld = pel.ldofs_own_only
        push!(buff_Phis, P[ld, :])
    end
    Kr_ff_factor = lu(Kr_ff)
    buffPp = fill(zero(eltype(Kr_ff_factor)), nr)
    buffKiPp = fill(zero(eltype(Kr_ff_factor)), nr)
    return TwoLevelPreConditioner(ddcomm, n, buff_Phis, Kr_ff_factor, buffPp, buffKiPp)
end

function (pre::TwoLevelPreConditioner)(q::PV, p::PV) where {PV<:PartitionedVector}
    partition_list = p.ddcomm.partition_list
    rhs_update_xt!(p)
    pre.buffPp .= zero(eltype(pre.buffPp))
    for i in eachindex(partition_list)
        ld = partition_list[i].entity_list.nonshared.ldofs_own_only
        pre.buffPp .+= pre.buff_Phis[i]' * p.buffers[i].ns[ld]
    end
    pre.buffKiPp .= pre.Kr_ff_factor \ pre.buffPp
    for i in eachindex(partition_list)
        q.buffers[i].ns .= 0
        ld = partition_list[i].entity_list.nonshared.ldofs_own_only
        q.buffers[i].ns[ld] .= pre.buff_Phis[i] * pre.buffKiPp
    end
    _lhs_update!(q)
    for i in eachindex(partition_list)
        q.buffers[i].xt .= partition_list[i].Kxt_ff_factor \ p.buffers[i].xt
    end
    lhs_update_xt!(q)
    q
end

function rhs_update_xt!(p::PV) where {PV<:PartitionedVector}
    partition_list = p.ddcomm.partition_list
    for i in eachindex(partition_list)
        pin = partition_list[i].entity_list.extended
        ld = pin.ldofs_own_only
        p.buffers[i].xt .= 0
        p.buffers[i].xt[ld] .= p.buffers[i].ns[ld]
    end
    for i in eachindex(partition_list)
        ldofs_other = partition_list[i].entity_list.extended.ldofs_other
        for j in eachindex(ldofs_other)
            if !isempty(ldofs_other[j])
                ldofs_self = partition_list[j].entity_list.extended.ldofs_self
                p.buffers[i].xt[ldofs_other[j]] .= p.buffers[j].xt[ldofs_self[i]]
            end
        end
    end
end

function lhs_update_xt!(q::PV) where {PV<:PartitionedVector}
    partition_list = q.ddcomm.partition_list
    for i in eachindex(partition_list)
        ldofs_self = partition_list[i].entity_list.extended.ldofs_self
        for j in eachindex(ldofs_self)
            if !isempty(ldofs_self[j])
                ldofs_other = partition_list[j].entity_list.extended.ldofs_other
                q.buffers[i].xt[ldofs_self[j]] .+= q.buffers[j].xt[ldofs_other[i]]
            end
        end
        qin = partition_list[i].entity_list.nonshared
        ld = qin.ldofs_own_only
        q.buffers[i].ns[ld] .+= q.buffers[i].xt[ld]
    end
end

function rhs(ddcomm::DDC) where {DDC<:DDCoNCSeqComm{PD} where {PD<:CoNCPartitionData}}
    rhs = deepcopy(ddcomm.partition_list[1].rhs)
    rhs .= 0
    for i in eachindex(ddcomm.partition_list)
        rhs .+= ddcomm.partition_list[i].rhs
    end  
    return rhs
end

end # module DDCoNCSeqModule
