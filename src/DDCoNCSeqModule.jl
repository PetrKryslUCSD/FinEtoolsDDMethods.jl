"""
    DDCoNCSeqModule  

Module for operations on partitions of finite element models for CG solves based
on the Coherent Nodal Clusters.

Implementation for sequential execution.

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
`partition_list[i].own.ldofs_other[j]` are degrees
of freedom that partition `i` receives from partition `j`, and
`partition_list[j].own.ldofs_self[i]` are degrees of
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

torank(i) = i - 1

"""
    DDCoNCSeqComm{PD<:CoNCPartitionData}

Communicator for sequential execution.
"""
struct DDCoNCSeqComm{PD<:CoNCPartitionData}
    partition_list::Vector{PD}
    gdim::Int
end

function DDCoNCSeqComm(comm, cpi, fes, make_matrix, make_interior_load)
    partition_list = _make_partitions(cpi, fes, make_matrix, make_interior_load)
    gdim = nfreedofs(cpi.u)
    return DDCoNCSeqComm(partition_list, gdim)
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
    ownv::Vector{T}
    extv::Vector{T}
    vrecvv::Vector{Vector{T}}
    vsendv::Vector{Vector{T}}
end

struct PartitionedVector{DDC<:DDCoNCSeqComm{PD} where {PD<:CoNCPartitionData}, T<:Number}
    ddcomm::DDC
    buffers::Vector{_Buffers{T}}
end

function _make_buffers(::Type{T}, partition_list) where {T}
    buffers = _Buffers{T}[]
    for partition in partition_list
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
        push!(buffers, _Buffers(ownv, extv, vrecvv, vsendv))
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
        a.buffers[i].ownv .= v   
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
        a.buffers[i].ownv .= v[partition_list[i].entity_list.own.global_dofs]
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
        ownd = el.own.global_dofs[1:el.own.num_own_dofs]
        lod = el.own.dof_glob2loc[ownd]
        v[ownd] .= a.buffers[i].ownv[lod]
    end
    v
end

"""
    vec_collect(a::PV) where {PV<:PartitionedVector}

Collect the partitioned vector into a global vector.
"""
function vec_collect(a::PV) where {PV<:PartitionedVector}
    v = zeros(eltype(a.buffers[1].ownv), a.ddcomm.gdim)
    return vec_copyto!(v, a)
end

"""
    vec_copyto!(y::PV, x::PV) where {PV<:PartitionedVector}

Copy one partitioned vector to another.

The contents of `x` is copied into `y`.
"""
function vec_copyto!(y::PV, x::PV) where {PV<:PartitionedVector}
    partition_list = y.ddcomm.partition_list
    for i in eachindex(partition_list)
        @. y.buffers[i].ownv = x.buffers[i].ownv
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

"""
    vec_aypx!(y::PV, a, x::PV) where {PV<:PartitionedVector}

Compute `y = a y + x`.
"""
function vec_aypx!(y::PV, a, x::PV) where {PV<:PartitionedVector}
    partition_list = y.ddcomm.partition_list
    for i in eachindex(partition_list)
        @. y.buffers[i].ownv = a * y.buffers[i].ownv + x.buffers[i].ownv
    end
    y
end

"""
    vec_ypax!(y::PV, a, x::PV) where {PV<:PartitionedVector}

Compute `y = y + a x`.
"""
function vec_ypax!(y::PV, a, x::PV) where {PV<:PartitionedVector}
    partition_list = y.ddcomm.partition_list
    for i in eachindex(partition_list)
        @. y.buffers[i].ownv = y.buffers[i].ownv + a * x.buffers[i].ownv
    end
    y
end

"""
    vec_dot(x::PV, y::PV) where {PV<:PartitionedVector}

Compute the dot product of two partitioned vectors.
"""
function vec_dot(x::PV, y::PV) where {PV<:PartitionedVector}
    partition_list = y.ddcomm.partition_list
    result = zero(eltype(x.buffers[1].ownv))
    for i in eachindex(partition_list)
        own = 1:partition_list[i].entity_list.own.num_own_dofs
        result += dot(y.buffers[i].ownv[own], x.buffers[i].ownv[own])
    end
    return result
end

function _rhs_update!(p::PV) where {PV<:PartitionedVector}
    partition_list = p.ddcomm.partition_list
    # Start all sends
    for self in eachindex(partition_list)
        ldofs_self = partition_list[self].entity_list.own.ldofs_self
        bs = p.buffers[self]
        for other in eachindex(ldofs_self)
            if !isempty(ldofs_self[other])
                n = length(ldofs_self[other])
                bs.vsendv[other][1:n] .= bs.ownv[ldofs_self[other]]
            end
        end
    end
    # Start all receives
    for self in eachindex(partition_list)
        ldofs_other = partition_list[self].entity_list.own.ldofs_other
        bs = p.buffers[self]
        for other in eachindex(ldofs_other)
            bo = p.buffers[other]
            if !isempty(ldofs_other[other])
                n = length(ldofs_other[other])
                bs.vrecvv[other][1:n] .= bo.vsendv[self][1:n]
            end
        end
    end
    # Wait for all receives
    # Unpack from buffers
    for self in eachindex(partition_list)
        ldofs_other = partition_list[self].entity_list.own.ldofs_other
        bs = p.buffers[self]
        for other in eachindex(ldofs_other)
            if !isempty(ldofs_other[other])
                n = length(ldofs_other[other])
                bs.ownv[ldofs_other[other]] .= bs.vrecvv[other][1:n]
            end
        end
    end
end

function _lhs_update!(q::PV) where {PV<:PartitionedVector}
    partition_list = q.ddcomm.partition_list
    # Start all sends
    for self in eachindex(partition_list)
        ldofs_other = partition_list[self].entity_list.own.ldofs_other
        bs = q.buffers[self]
        for other in eachindex(ldofs_other)
            if !isempty(ldofs_other[other])
                n = length(ldofs_other[other])
                bs.vsendv[other][1:n] .= bs.ownv[ldofs_other[other]]
            end
        end
    end
    # Start all receives
    for self in eachindex(partition_list)
        ldofs_self = partition_list[self].entity_list.own.ldofs_self
        bs = q.buffers[self]
        for other in eachindex(ldofs_self)
            bo = q.buffers[other]
            if !isempty(ldofs_self[other])
                n = length(ldofs_self[other])
                bs.vrecvv[other][1:n] .= bo.vsendv[self][1:n]
            end
        end
    end
    # Wait for all receives
    # Unpack from buffers
    for self in eachindex(partition_list)
        ldofs_self = partition_list[self].entity_list.own.ldofs_self
        bs = q.buffers[self]
        for other in eachindex(ldofs_self)
            if !isempty(ldofs_self[other])
                n = length(ldofs_self[other])
                bs.ownv[ldofs_self[other]] .+= bs.vrecvv[other][1:n]
            end
        end
    end
end

function aop!(q::PV, p::PV) where {PV<:PartitionedVector}
    partition_list = p.ddcomm.partition_list
    _rhs_update!(p)
    for i in eachindex(partition_list)
        q.buffers[i].ownv .= partition_list[i].Kown_ff * p.buffers[i].ownv
    end
    _lhs_update!(q)
    q    
end

struct TwoLevelPreConditioner{DDC<:DDCoNCSeqComm{PD} where {PD<:CoNCPartitionData}, T, IT, FACTOR}
    ddcomm::DDC
    n::IT
    buff_Phis::Vector{SparseMatrixCSC{T, IT}}
    Kr_ff::SparseMatrixCSC{T, IT} # DEBUG
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
        pel = partition_list[i].entity_list.own
        # First we work with all the degrees of freedom on the partition
        P = Phi[pel.global_dofs, :]
        Kr_ff += (P' * partition_list[i].Kown_ff * P)
        # the transformation matrices are now resized to only the own dofs
        ld = pel.ldofs_own_only
        push!(buff_Phis, P[ld, :])
    end
    Kr_ff_factor = lu(Kr_ff)
    buffPp = fill(zero(eltype(Kr_ff_factor)), nr)
    buffKiPp = fill(zero(eltype(Kr_ff_factor)), nr)
    return TwoLevelPreConditioner(ddcomm, n, buff_Phis, Kr_ff, Kr_ff_factor, buffPp, buffKiPp)
end

function (pre::TwoLevelPreConditioner)(q::PV, p::PV) where {PV<:PartitionedVector}
    partition_list = p.ddcomm.partition_list
    _rhs_update_xt!(p)
    pre.buffPp .= zero(eltype(pre.buffPp))
    for self in eachindex(partition_list)
        ld = partition_list[self].entity_list.own.ldofs_own_only
        pre.buffPp .+= pre.buff_Phis[self]' * p.buffers[self].ownv[ld]
    end
    pre.buffKiPp .= pre.Kr_ff_factor \ pre.buffPp
    for self in eachindex(partition_list)
        q.buffers[self].ownv .= 0
        ld = partition_list[self].entity_list.own.ldofs_own_only
        q.buffers[self].ownv[ld] .= pre.buff_Phis[self] * pre.buffKiPp
    end
    _lhs_update!(q)
    for self in eachindex(partition_list)
        q.buffers[self].extv .= partition_list[self].Kxt_ff_factor \ p.buffers[self].extv
    end
    _lhs_update_xt!(q)
    q
end

function _rhs_update_xt!(p::PV) where {PV<:PartitionedVector}
    partition_list = p.ddcomm.partition_list
    # Copy data from owned to extended
    for self in eachindex(partition_list)
        psx = partition_list[self].entity_list.extended
        p.buffers[self].extv .= 0
        ld = psx.ldofs_own_only
        p.buffers[self].extv[ld] .= p.buffers[self].ownv[ld]
    end
    # Start all sends
    for self in eachindex(partition_list)
        ldofs_self = partition_list[self].entity_list.extended.ldofs_self
        bs = p.buffers[self]
        for other in eachindex(ldofs_self)
            if !isempty(ldofs_self[other])
                n = length(ldofs_self[other])
                bs.vsendv[other][1:n] .= bs.extv[ldofs_self[other]]
            end
        end
    end
    # Start all receives
    for self in eachindex(partition_list)
        ldofs_other = partition_list[self].entity_list.extended.ldofs_other
        bs = p.buffers[self]
        for other in eachindex(ldofs_other)
            bo = p.buffers[other]
            if !isempty(ldofs_other[other])
                n = length(ldofs_other[other])
                bs.vrecvv[other][1:n] .= bo.vsendv[self][1:n]
            end
        end
    end
    # Wait for all receives
    # Unpack from buffers
    for self in eachindex(partition_list)
        ldofs_other = partition_list[self].entity_list.extended.ldofs_other
        bs = p.buffers[self]
        for other in eachindex(ldofs_other)
            if !isempty(ldofs_other[other])
                n = length(ldofs_other[other])
                bs.extv[ldofs_other[other]] .= bs.vrecvv[other][1:n]
            end
        end
    end
end

function _lhs_update_xt!(q::PV) where {PV<:PartitionedVector}
    partition_list = q.ddcomm.partition_list
    # Start all sends
    for self in eachindex(partition_list)
        ldofs_other = partition_list[self].entity_list.extended.ldofs_other
        bs = q.buffers[self]
        for other in eachindex(ldofs_other)
            if !isempty(ldofs_other[other])
                n = length(ldofs_other[other])
                bs.vsendv[other][1:n] .= bs.extv[ldofs_other[other]]
            end
        end
    end
    # Start all receives
    for self in eachindex(partition_list)
        ldofs_self = partition_list[self].entity_list.extended.ldofs_self
        bs = q.buffers[self]
        for other in eachindex(ldofs_self)
            bo = q.buffers[other]
            if !isempty(ldofs_self[other])
                n = length(ldofs_self[other])
                bs.vrecvv[other][1:n] .= bo.vsendv[self][1:n]
            end
        end
    end
    # Wait for all receives
    # Unpack from buffers
    for self in eachindex(partition_list)
        ldofs_self = partition_list[self].entity_list.extended.ldofs_self
        bs = q.buffers[self]
        for other in eachindex(ldofs_self)
            if !isempty(ldofs_self[other])
                n = length(ldofs_self[other])
                bs.extv[ldofs_self[other]] .+= bs.vrecvv[other][1:n]
            end
        end
        ld = partition_list[self].entity_list.own.ldofs_own_only
        q.buffers[self].ownv[ld] .+= q.buffers[self].extv[ld]
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
