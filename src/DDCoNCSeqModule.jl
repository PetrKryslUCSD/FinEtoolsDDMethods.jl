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
    partition_list  = [CoNCPartitionData(cpi) for i in 1:npartitions(cpi)]
    for i in eachindex(partition_list)
        partition_list[i] = CoNCPartitionData(cpi, i, fes, make_matrix, make_interior_load)
    end  
    return partition_list
end

function _a_mult!(q, cpi, partition_list, timers, p)
    q .= zero(eltype(q))
    for partition in partition_list
        d = partition.nsdof
        partition.ntempp .= p[d]
        mul!(partition.ntempq, partition.nonshared_K, partition.ntempp)
        q[d] .+= partition.ntempq
    end
    q
end

mutable struct AOperator{PD, CI, TD}
    partition_list::Vector{PD}
    cpi::CI
    timers::TD
end

function AOperator(partition_list, cpi) 
    AOperator(partition_list, cpi,
        (rank == 0
         ? set_up_timers("1_send_nbuffs", "2_add_nbuffs", "3_total")
         : set_up_timers("1_recv", "2_mult_local", "3_total"))
    )
end

function a_mult!(q, aop::A, p) where {A<:AOperator}
    _a_mult!(q, aop.cpi, aop.partition_list, aop.timers, p)
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
