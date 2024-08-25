"""
    DDCoNCMPIModule  

Module for operations on partitions of finite element models for solves based
on the Coherent Nodal Clusters.
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
import Base: size, eltype
import LinearAlgebra: mul!, eigen
using Statistics: mean
using ..FENodeToPartitionMapModule: FENodeToPartitionMap
using ShellStructureTopo

using ..PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData

function partition_multiply!(q, partition, p)
    q .= zero(eltype(q))
    if partition !== nothing
        d = partition.ndof
        partition.ntempp .= p[d]
        mul!(partition.ntempq, partition.nonoverlapping_K, partition.ntempp)
        q[d] .+= partition.ntempq
    end
    q
end

function precondition_global_solve!(q, Krfactor, Phi, p) 
    q .= Phi * (Krfactor \ (Phi' * p))
    q
end

function precondition_local_solve!(q, partition, p) 
    q .= zero(eltype(q))
    if partition !== nothing
        d = partition.odof
        partition.otempp .= p[d]
        ldiv!(partition.otempq, partition.overlapping_K_factor, partition.otempp)
        q[d] .+= partition.otempq
    end
    q
end

end # module DDCoNCMPIModule
