"""
    DDCoNCSeqModule  

Module for operations on partitions of finite element models for CG solves based
on the Coherent Nodal Clusters.

Implementation for sequential execution.
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
import LinearAlgebra: mul!, eigen
using Statistics: mean
using ..FENodeToPartitionMapModule: FENodeToPartitionMap
using ShellStructureTopo

using ..PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData, npartitions, NONSHARED, EXTENDED
using ..FinEtoolsDDMethods: set_up_timers, update_timer!, reset_timers!

import ..CGModule: vec_copyto!
import ..CGModule: vec_aypx!
import ..CGModule: vec_ypax!
import ..CGModule: vec_dot

function make_partitions(cpi, fes, make_matrix, make_interior_load)
    partition_list  = [CoNCPartitionData(cpi, i) for i in 1:npartitions(cpi)]
    for i in eachindex(partition_list)
        partition_list[i] = CoNCPartitionData(cpi, i, fes, make_matrix, make_interior_load)
    end  
    return partition_list
end

struct PartitionedVector{PD<:CoNCPartitionData, T<:Number}
    partition_list::Vector{PD}
    buff_ns::Vector{Vector{T}}
    buff_xt::Vector{Vector{T}}
end

function PartitionedVector(::Type{T}, partition_list::Vector{PD}) where {T, PD<:CoNCPartitionData}
    buff_ns = [fill(zero(T), length(partition.entity_list[NONSHARED].global_dofs)) for partition in partition_list]
    buff_xt = [fill(zero(T), length(partition.entity_list[EXTENDED].global_dofs)) for partition in partition_list]
    return PartitionedVector(partition_list, buff_ns, buff_xt)
end

"""
    vec_copyto!(a::PV, v::T) where {PV<:PartitionedVector, T}

Copy a single value into the partitioned vector.
"""
function vec_copyto!(a::PV, v::T) where {PV<:PartitionedVector, T}
    for i in eachindex(a.partition_list)
        a.buff_ns[i] .= v   
    end
    a
end

"""
    vec_copyto!(a::PV, v::Vector{T}) where {PV<:PartitionedVector, T}

Copy a global vector into a partitioned vector.
"""
function vec_copyto!(a::PV, v::Vector{T}) where {PV<:PartitionedVector, T}
    for i in eachindex(a.partition_list)
        a.buff_ns[i] .= v[a.partition_list[i].entity_list[NONSHARED].global_dofs]
    end
    a
end

"""
    vec_copyto!(v::Vector{T}, a::PV) where {PV<:PartitionedVector, T}

Copy a partitioned vector into a global vector.
"""
function vec_copyto!(v::Vector{T}, a::PV) where {PV<:PartitionedVector, T}
    for i in eachindex(a.partition_list)
        el = a.partition_list[i].entity_list
        ownd = el[NONSHARED].global_dofs[1:el[NONSHARED].num_own_dofs]
        lod = el[NONSHARED].global_to_local[ownd]
        v[ownd] .= a.buff_ns[i][lod]
    end
    a
end

"""
    vec_copyto!(y::PV, x::PV) where {PV<:PartitionedVector}

Copy one partitioned vector to another.

The contents of `x` is copied into `y`.
"""
function vec_copyto!(y::PV, x::PV) where {PV<:PartitionedVector}
    for i in eachindex(y.partition_list)
        @. y.buff_ns[i] = x.buff_ns[i]
    end
    y
end

"""
    deepcopy(a::PV) where {PV<:PartitionedVector}

Create a deep copy of a partitioned vector.
"""
function deepcopy(a::PV) where {PV<:PartitionedVector}
    return PartitionedVector(a.partition_list, deepcopy(a.buff_ns), deepcopy(a.buff_xt))
end

# Computes y = x + a y.
function vec_aypx!(y::PV, a, x::PV) where {PV<:PartitionedVector}
    for i in eachindex(y.partition_list)
        @. y.buff_ns[i] = a * y.buff_ns[i] + x.buff_ns[i]
    end
    y
end

# Computes y += a x
function vec_ypax!(y::PV, a, x::PV) where {PV<:PartitionedVector}
    for i in eachindex(y.partition_list)
        @. y.buff_ns[i] = y.buff_ns[i] + a * x.buff_ns[i]
    end
    y
end

function vec_dot(x::PV, y::PV) where {PV<:PartitionedVector}
    result = zero(eltype(x.buff_ns[1]))
    for i in eachindex(y.partition_list)
        own = 1:y.partition_list[i].entity_list[NONSHARED].num_own_dofs
        result += dot(y.buff_ns[i][own], x.buff_ns[i][own])
    end
    return result
end

function rhs_update!(p::PV) where {PV<:PartitionedVector}
    for i in eachindex(p.partition_list)
        pin = p.partition_list[i].entity_list[NONSHARED]
        for j in eachindex(pin.local_receive_dofs)
            if !isempty(pin.local_receive_dofs[j])
                @assert i != j
                pjn = p.partition_list[j].entity_list[NONSHARED]
                ldi = pin.local_receive_dofs[j]
                ldj = pjn.local_send_dofs[i]
                p.buff_ns[i][ldi] .= p.buff_ns[j][ldj]
            end
        end
    end
end

function lhs_update!(q::PV) where {PV<:PartitionedVector}
    for i in eachindex(q.partition_list)
        qin = q.partition_list[i].entity_list[NONSHARED]
        for j in eachindex(qin.local_send_dofs)
            if !isempty(qin.local_send_dofs[j])
                @assert i != j
                qjn = q.partition_list[j].entity_list[NONSHARED]
                ldi = qin.local_send_dofs[j]
                ldj = qjn.local_receive_dofs[i]
                q.buff_ns[i][ldi] .+= q.buff_ns[j][ldj]
            end
        end
    end
end

function aop!(q::PV, p::PV) where {PV<:PartitionedVector}
    rhs_update!(p)
    for i in eachindex(q.partition_list)
        q.buff_ns[i] .= p.partition_list[i].Kns_ff * p.buff_ns[i]
    end
    lhs_update!(q)
    q    
end

struct TwoLevelPreConditioner{PD<:CoNCPartitionData, T, IT, FACTOR}
    partition_list::Vector{PD}
    n::IT
    buff_Phis::Vector{SparseMatrixCSC{T, IT}}
    Kr_ff_factor::FACTOR
    buffq::Vector{T}
    buffPp::Vector{T}
    buffKiPp::Vector{T}
end

function TwoLevelPreConditioner(partition_list, Phi)
    n = size(Phi, 1)
    nr = size(Phi, 2)
    buff_Phis = typeof(Phi)[]
    sizehint!(buff_Phis, length(partition_list))
    Kr_ff = spzeros(nr, nr)
    for i in eachindex(partition_list)
        pel = partition_list[i].entity_list[NONSHARED]
        # First we work with all the degrees of freedom on the partition
        P = Phi[pel.global_dofs, :]
        Kr_ff += (P' * partition_list[i].Kns_ff * P)
        # the transformation matrices are now resized to only the own dofs
        ld = pel.local_own_dofs
        push!(buff_Phis, P[ld, :])
    end
    Kr_ff_factor = lu(Kr_ff)
    buffq = fill(zero(eltype(Kr_ff_factor)), n)
    buffPp = fill(zero(eltype(Kr_ff_factor)), nr)
    buffKiPp = fill(zero(eltype(Kr_ff_factor)), nr)
    return TwoLevelPreConditioner(partition_list, n, buff_Phis, Kr_ff_factor, buffq, buffPp, buffKiPp)
end

function (pre::TwoLevelPreConditioner)(q::PV, p::PV) where {PV<:PartitionedVector}
    # vec_copyto!(q, p) # this would be a identity preconditioner 
    rhs_update!(p)
    rhs_update_xt!(p)
    pre.buffPp .= zero(eltype(pre.buffPp))
    for i in eachindex(q.partition_list)
        ld = q.partition_list[i].entity_list[NONSHARED].local_own_dofs
        pre.buffPp .+= pre.buff_Phis[i]' * p.buff_ns[i][ld]
    end
    pre.buffKiPp .= pre.Kr_ff_factor \ pre.buffPp
    for i in eachindex(q.partition_list)
        ld = q.partition_list[i].entity_list[NONSHARED].local_own_dofs
        q.buff_ns[i] .= 0
        q.buff_ns[i][ld] .= pre.buff_Phis[i] * pre.buffKiPp
    end
    lhs_update!(q)
    for i in eachindex(q.partition_list)
        q.buff_xt[i] .= p.partition_list[i].Kxt_ff_factor \ p.buff_xt[i]
    end
    # lhs_update_xt!(q)
    q
end

function rhs_update_xt!(p::PV) where {PV<:PartitionedVector}
    for i in eachindex(p.partition_list)
        pin = p.partition_list[i].entity_list[EXTENDED]
        ld = pin.local_own_dofs
        p.buff_xt[i] .= 0
        p.buff_xt[i][ld] .= p.buff_ns[i][ld]
    end
    for i in eachindex(p.partition_list)
        pin = p.partition_list[i].entity_list[EXTENDED]
        for j in eachindex(pin.local_receive_dofs)
            if !isempty(pin.local_receive_dofs[j])
                @assert i != j
                pjn = p.partition_list[j].entity_list[EXTENDED]
                ldi = pin.local_receive_dofs[j]
                ldj = pjn.local_send_dofs[i]
                p.buff_xt[i][ldi] .= p.buff_xt[j][ldj]
            end
        end
    end
end

function lhs_update_xt!(q::PV) where {PV<:PartitionedVector}
    for i in eachindex(q.partition_list)
        qie = q.partition_list[i].entity_list[EXTENDED]
        for j in eachindex(qie.local_send_dofs)
            if !isempty(qie.local_send_dofs[j])
                @assert i != j
                qjn = q.partition_list[j].entity_list[EXTENDED]
                ldi = qie.local_send_dofs[j]
                ldj = qjn.local_receive_dofs[i]
                q.buff_xt[i][ldi] .+= q.buff_xt[j][ldj]
            end
        end
        qin = q.partition_list[i].entity_list[NONSHARED]
        ld = qin.local_own_dofs
        q.buff_ns[i][ld] .+= q.buff_xt[i][ld]
    end
end

end # module DDCoNCSeqModule
