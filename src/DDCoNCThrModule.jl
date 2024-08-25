"""
    DDCoNCThrModule  

Module for operations on partitions of finite element models for solves based
on the Coherent Nodal Clusters.
"""
module DDCoNCThrModule

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

using ..PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData

function partition_multiply!(q, partition_list, p)
    q .= zero(eltype(q))
    Threads.@threads for partition in partition_list
        d = partition.ndof
        partition.ntempp .= p[d]
        mul!(partition.ntempq, partition.nonoverlapping_K, partition.ntempp)
    end
    for partition in partition_list
        d = partition.ndof
        q[d] .+= partition.ntempq
    end
    q
end

function _precondition_global_solve!(q, Krfactor, Phi, p) 
    q .+= Phi * (Krfactor \ (Phi' * p))
    q
end

function _precondition_local_solve!(q, partition_list, p) 
    Threads.@threads for partition in partition_list
        d = partition.odof
        partition.otempp .= p[d]
        ldiv!(partition.otempq, partition.overlapping_K_factor, partition.otempp)
    end
    for partition in partition_list
        d = partition.odof
        q[d] .+= partition.otempq
    end
    q
end

function preconditioner!(Krfactor, Phi, partition_list)
    function M!(q, p)
        q .= zero(eltype(q))
        _precondition_local_solve!(q, partition_list, p)
        _precondition_global_solve!(q, Krfactor, Phi, p)
        q
    end
    return M!
end

end # module DDCoNCThrModule
