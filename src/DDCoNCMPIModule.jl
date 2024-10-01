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
using MPI

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

function partition_mult!(q, cpi, comm, rank, partition, p)
    q .= zero(eltype(q))
    if rank == 0
        requests = MPI.Request[]
        for i in 1:length(cpi.dof_lists)
            d = cpi.dof_lists[i].nonoverlapping
            cpi.nbuffs[i] .= p[d]
            req = MPI.Isend(cpi.nbuffs[i], comm; dest = i)
            push!(requests, req)
        end
        MPI.Waitall!(requests)
        requests = [MPI.Irecv!(cpi.nbuffs[i], comm; source = i) for i in 1:length(cpi.dof_lists)]
        while true
            i = MPI.Waitany(requests)
            if i === nothing
                break
            end
            d = cpi.dof_lists[i].nonoverlapping
            q[d] .+= cpi.nbuffs[i]
        end
    else
        d = partition.ndof
        if size(partition.nonoverlapping_K,1) != length(d) # trim size
            partition.nonoverlapping_K = partition.nonoverlapping_K[d, d]
        end
        MPI.Recv!(partition.ntempp, comm; source=0)
        mul!(partition.ntempq, partition.nonoverlapping_K, partition.ntempp)
        MPI.Send(partition.ntempq, comm; dest = 0)
    end
    q
end

function precond_local!(q, cpi, comm, rank, partition, p) 
    q .= zero(eltype(q))
    if rank == 0
        requests = MPI.Request[]
        for i in 1:length(cpi.dof_lists)
            d = cpi.dof_lists[i].overlapping
            cpi.obuffs[i] .= p[d]
            req = MPI.Isend(cpi.obuffs[i], comm; dest = i)
            push!(requests, req)
        end
        MPI.Waitall!(requests)
        requests = [MPI.Irecv!(cpi.obuffs[i], comm; source = i) for i in 1:length(cpi.dof_lists)]
        while true
            i = MPI.Waitany(requests)
            if i === nothing
                break
            end
            d = cpi.dof_lists[i].overlapping
            q[d] .+= cpi.obuffs[i]
        end
    else
        MPI.Recv!(partition.otempp, comm; source = 0)
        ldiv!(partition.otempq, partition.overlapping_K_factor, partition.otempp)
        MPI.Send(partition.otempq, comm; dest = 0)
    end
    q
end

function precond_2level!(q, Krfactor, Phi, cpi, comm, rank, partition, p) 
    q .= zero(eltype(q))
    if rank == 0
        requests = MPI.Request[]
        for i in 1:length(cpi.dof_lists)
            d = cpi.dof_lists[i].overlapping
            cpi.obuffs[i] .= p[d]
            req = MPI.Isend(cpi.obuffs[i], comm; dest = i)
            push!(requests, req)
        end
        q .= Phi * (Krfactor \ (Phi' * p))
        MPI.Waitall!(requests)
        requests = [MPI.Irecv!(cpi.obuffs[i], comm; source = i) for i in 1:length(cpi.dof_lists)]
        while true
            i = MPI.Waitany(requests)
            if i === nothing
                break
            end
            d = cpi.dof_lists[i].overlapping
            q[d] .+= cpi.obuffs[i]
        end
    else
        MPI.Recv!(partition.otempp, comm; source = 0)
        ldiv!(partition.otempq, partition.overlapping_K_factor, partition.otempp)
        MPI.Send(partition.otempq, comm; dest = 0)
    end
    q
end

end # module DDCoNCMPIModule
