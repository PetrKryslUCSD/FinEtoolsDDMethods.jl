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
        if size(partition.nonoverlapping_K,1) != length(d) # trim size
            partition.nonoverlapping_K = partition.nonoverlapping_K[d, d]
        end
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

function scatter_nvector(v, cpi, comm, rank, partition)
    @show "scatter_nvector"
    if rank == 0
        for i in 1:length(cpi.dof_lists)
            d = cpi.dof_lists[i].nonoverlapping
            MPI.Send(v[d], comm; dest = i)
        end
    else
        d = cpi.dof_lists[rank].nonoverlapping
        MPI.Recv!(partition.ntempp, comm; source = 0)
    end
end

function gather_nvector(v, cpi, comm, rank, partition)
    @show "gather_nvector"
    if rank == 0
        for i in 1:length(cpi.dof_lists)
            MPI.Recv!(cpi.nbuffs[i], comm; source = i)
            v[d] .+= cpi.nbuffs[i]
        end
    else
        MPI.Send(partition.ntempq, comm; dest = i)
    end
end

function partition_mult!(q, cpi, comm, rank, partition, p)
    scatter_nvector(p, cpi, comm, rank, partition)
    if rank == 0
        q .= zero(eltype(q))
    else
        d = partition.ndof
        if size(partition.nonoverlapping_K,1) != length(d) # trim size
            partition.nonoverlapping_K = partition.nonoverlapping_K[d, d]
        end
        mul!(partition.ntempq, partition.nonoverlapping_K, partition.ntempp)
    end
    gather_nvector(q, cpi, comm, rank, partition)
    q
end

end # module DDCoNCMPIModule
