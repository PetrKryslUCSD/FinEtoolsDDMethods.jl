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

using ..PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData, npartitions
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

struct PartitionedVector{PD, T}
    partition_list::Vector{PD}
    buff_ns::Vector{Vector{T}}
    buff_xt::Vector{Vector{T}}
end

function PartitionedVector(::Type{T}, partition_list::Vector{PD}) where {T, PD<:CoNCPartitionData}
    nonshared = [fill(zero(T), length(partition.entity_list.nonshared.global_dofs)) for partition in partition_list]
    extended = [fill(zero(T), length(partition.entity_list.extended.global_dofs)) for partition in partition_list]
    return PartitionedVector(partition_list, nonshared, extended)
end

function vec_copyto!(a::PV, v::T) where {PV<:PartitionedVector, T}
    for i in eachindex(a.partition_list)
        od = a.partition_list[i].entity_list.nonshared.own_global_dofs
        lod = a.partition_list[i].entity_list.nonshared.global_to_local[od]
        a.buff_ns[i][lod] .= v   
    end
    a
end

function vec_copyto!(a::PV, v::Vector{T}) where {PV<:PartitionedVector, T}
    for i in eachindex(a.partition_list)
        od = a.partition_list[i].entity_list.nonshared.own_global_dofs
        lod = a.partition_list[i].entity_list.nonshared.global_to_local[od]
        a.buff_ns[i][lod] .= v[a.partition_list[i].entity_list.nonshared.own_global_dofs]
    end
    a
end

function vec_copyto!(v::Vector{T}, a::PV) where {PV<:PartitionedVector, T}
    for i in eachindex(a.partition_list)
        od = a.partition_list[i].entity_list.nonshared.own_global_dofs
        lod = a.partition_list[i].entity_list.nonshared.global_to_local[od]
        v[od] .= a.buff_ns[i][lod]
    end
    a
end

function deepcopy(a::PV) where {PV<:PartitionedVector}
    return PartitionedVector(a.partition_list, deepcopy(a.buff_ns), deepcopy(a.buff_xt))
end

function vec_copyto!(y::PV, x::PV) where {PV<:PartitionedVector}
    for i in eachindex(y.partition_list)
        @. y.buff_ns[i] = x.buff_ns[i]
    end
    y
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
        result += dot(y.buff_ns[i], x.buff_ns[i])
    end
    return result
end

function gather!(p::PV) where {PV<:PartitionedVector}
    for i in eachindex(p.partition_list)
        pin = p.partition_list[i].entity_list.nonshared
        for j in eachindex(pin.receive_dofs)
            if !isempty(pin.receive_dofs[j])
                @assert i != j
                pjn = p.partition_list[j].entity_list.nonshared
                ldi = pin.global_to_local[pin.receive_dofs[j]]
                ldj = pjn.global_to_local[pjn.send_dofs[i]]
                p.buff_ns[i][ldi] .= p.buff_ns[j][ldj]
            end
        end
    end
end

function scatter!(q::PV) where {PV<:PartitionedVector}
    for i in eachindex(q.partition_list)
        qin = q.partition_list[i].entity_list.nonshared
        for j in eachindex(qin.send_dofs)
            if !isempty(qin.send_dofs[j])
                @assert i != j
                qjn = q.partition_list[j].entity_list.nonshared
                ldi = qin.global_to_local[qin.send_dofs[j]]
                ldj = qjn.global_to_local[qjn.receive_dofs[i]]
                q.buff_ns[i][ldi] .+= q.buff_ns[j][ldj]
            end
        end
    end
end

function aop!(q::PV, p::PV) where {PV<:PartitionedVector}
    gather!(p)
    for i in eachindex(q.partition_list)
        q.buff_ns[i] .= p.partition_list[i].nonshared_K * p.buff_ns[i]
    end
    scatter!(q)
    q    
end

function _precondition_global_solve!(q, Krfactor, Phi, p) 
    q .+= Phi * (Krfactor \ (Phi' * p))
    q
end

function _precondition_local_solve!(q, partition_list, p) 
    for partition in partition_list
        d = partition.alldof
        partition.otempp .= p[d]
        ldiv!(partition.otempq, partition.all_K_factor, partition.otempp)
        q[d] .+= partition.otempq
    end
    q
end

function preconditioner!(Krfactor, Phi, partition_list)
    # Make sure the partitions only refer to the local non-overlapping dofs
    for _p in partition_list
        _p.nonshared_K = _p.nonshared_K[_p.nsdof, _p.nsdof]
    end
    function M!(q, p)
        q .= zero(eltype(q))
        _precondition_local_solve!(q, partition_list, p)
        _precondition_global_solve!(q, Krfactor, Phi, p)
        q
    end
    return M!
end

end # module DDCoNCSeqModule
