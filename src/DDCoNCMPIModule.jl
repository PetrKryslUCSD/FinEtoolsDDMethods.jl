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
using ..CoNCUtilitiesModule: conc_cache
using ..FinEtoolsDDMethods: set_up_timers, update_timer!, reset_timers!

function partition_mult!(q, cpi, comm, rank, partition, timers, p)
    if rank == 0
        tstart0 = MPI.Wtime()
        q .= zero(eltype(q))
        tstart = MPI.Wtime()
        requests = MPI.Request[]
        for i in 1:length(cpi.dof_lists)
            d = cpi.dof_lists[i].nonoverlapping
            cpi.nbuffs[i] .= p[d]
            req = MPI.Isend(cpi.nbuffs[i], comm; dest = i)
            push!(requests, req)
        end
        # MPI.Waitall!(requests) # This may not be needed?
        update_timer!(timers, "1_send_nbuffs", MPI.Wtime() - tstart)
        tstart = MPI.Wtime()
        requests = [MPI.Irecv!(cpi.nbuffs[i], comm; source = i) for i in 1:length(cpi.dof_lists)]
        while true
            i = MPI.Waitany(requests)
            if i === nothing
                break
            end
            d = cpi.dof_lists[i].nonoverlapping
            q[d] .+= cpi.nbuffs[i]
        end
        update_timer!(timers, "2_add_nbuffs", MPI.Wtime() - tstart)
        update_timer!(timers, "3_total", MPI.Wtime() - tstart0)
    else
        d = partition.ndof
        if size(partition.nonoverlapping_K, 1) != length(d) # trim size
            partition.nonoverlapping_K = partition.nonoverlapping_K[d, d]
        end
        tstart0 = MPI.Wtime()
        tstart = MPI.Wtime()
        MPI.Recv!(partition.ntempp, comm; source=0)
        update_timer!(timers, "1_recv", MPI.Wtime() - tstart)
        tstart = MPI.Wtime()
        mul!(partition.ntempq, partition.nonoverlapping_K, partition.ntempp)
        update_timer!(timers, "2_mult_local", MPI.Wtime() - tstart)
        MPI.Isend(partition.ntempq, comm; dest = 0)
        update_timer!(timers, "3_total", MPI.Wtime() - tstart0)
    end
    q
end

mutable struct MPIAOperator{PD, CI, TD}
    comm::MPI.Comm
    rank::Int
    partition::PD
    cpi::CI
    timers::TD
end

function MPIAOperator(comm, rank, partition, cpi) 
    MPIAOperator(comm, rank, partition, cpi,
        (rank == 0
         ? set_up_timers("1_send_nbuffs", "2_add_nbuffs", "3_total")
         : set_up_timers("1_recv", "2_mult_local", "3_total"))
    )
end

function partition_mult!(q, aop::A, p) where {A<:MPIAOperator}
    partition_mult!(q, aop.cpi, aop.comm, aop.rank, aop.partition, aop.timers, p)
end

function precond_2level!(q, cc, cpi, comm, rank, partition, timers, p) 
    q .= zero(eltype(q))
    if rank == 0
        tstart0 = MPI.Wtime()
        tstart = MPI.Wtime()
        requests = MPI.Request[]
        for i in 1:length(cpi.dof_lists)
            d = cpi.dof_lists[i].overlapping
            cpi.obuffs[i] .= p[d]
            req = MPI.Isend(cpi.obuffs[i], comm; dest = i)
            push!(requests, req)
        end
        update_timer!(timers, "1_send_obuffs", MPI.Wtime() - tstart)
        # q .= Phi * (Krfactor \ (Phi' * p))
        tstart = MPI.Wtime()
        mul!(cc.PhiTp, cc.Phi', p)
        ldiv!(cc.KrfactorPhiTp, cc.Krfactor, cc.PhiTp)
        q .= mul!(cc.q, cc.Phi, cc.KrfactorPhiTp)
        update_timer!(timers, "2_solve_global", MPI.Wtime() - tstart)
        MPI.Waitall!(requests)
        update_timer!(timers, "3_wait_obuffs", MPI.Wtime() - tstart0)
        tstart = MPI.Wtime()
        requests = [MPI.Irecv!(cpi.obuffs[i], comm; source = i) for i in 1:length(cpi.dof_lists)]
        while true
            i = MPI.Waitany(requests)
            if i === nothing
                break
            end
            d = cpi.dof_lists[i].overlapping
            q[d] .+= cpi.obuffs[i]
        end
        update_timer!(timers, "4_add_obuffs", MPI.Wtime() - tstart)
        update_timer!(timers, "5_total", MPI.Wtime() - tstart0)
    else
        tstart = MPI.Wtime()
        MPI.Recv!(partition.otempp, comm; source = 0)
        ldiv!(partition.otempq, partition.overlapping_K_factor, partition.otempp)
        MPI.Isend(partition.otempq, comm; dest = 0)
        update_timer!(timers, "1_solve_local", MPI.Wtime() - tstart)
    end
    q
end

mutable struct MPITwoLevelPreconditioner{PD, CI, CC, TD}
    comm::MPI.Comm
    rank::Int
    partition::PD
    cpi::CI
    cc::CC
    timers::TD
end

function MPITwoLevelPreconditioner(comm, rank, partition, cpi, ccache)
    MPITwoLevelPreconditioner(comm, rank, partition, cpi, ccache, 
        (rank == 0 
        ? set_up_timers("1_send_obuffs", "2_solve_global", "3_wait_obuffs", "4_add_obuffs", "5_total")
        : set_up_timers("1_solve_local"))
    )
end 

function precond_2level!(q, pre::P, p)  where {P<:MPITwoLevelPreconditioner}
    precond_2level!(q, pre.cc, pre.cpi, pre.comm, pre.rank, pre.partition, pre.timers, p)
end

end # module DDCoNCMPIModule
