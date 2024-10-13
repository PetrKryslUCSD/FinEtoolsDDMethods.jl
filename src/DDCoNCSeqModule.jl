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
import Base: size, eltype
import LinearAlgebra: mul!, eigen
using Statistics: mean
using ..FENodeToPartitionMapModule: FENodeToPartitionMap
using ShellStructureTopo

using ..PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData, npartitions
using ..FinEtoolsDDMethods: set_up_timers, update_timer!, reset_timers!

function make_partitions(cpi, fes, make_matrix, make_interior_load)
    partition_list  = [CoNCPartitionData(cpi, i) for i in 1:npartitions(cpi)]
    for i in eachindex(partition_list)
        partition_list[i] = CoNCPartitionData(cpi, i, fes, make_matrix, make_interior_load)
    end  
    return partition_list
end

struct PartitionedVector{PD, T}
    partition_list::Vector{PD}
    nonshared::Vector{Vector{T}}
    extended::Vector{Vector{T}}
end

function PartitionedVector(T, partition_list::Vector{CoNCPartitionData})
    nonshared = [fill(zero(T), length(partition.nonshared.global_dofs)) for partition in partition_list]
    extended = [fill(zero(T), length(partition.extended.global_dofs)) for partition in partition_list]
    return PartitionedVector(partition_list, nonshared, extended)
end

function aop!(q::PV, p::PV) where {PV<:PartitionedVector}
    for i in eachindex(q.partition_list)
        
    end
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
