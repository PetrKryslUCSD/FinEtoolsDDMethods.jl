"""
    DDCoNCMPIModule  

Module for operations on partitions of finite element models for CG solves based
on the Coherent Nodal Clusters.

Implementation for MPI execution.

The module provides the `PartitionedVector` as a mechanism for working with
partitions. The partitioned vector defines a mechanism for storing and
communicating data for each partition. The data is stored in two buffers: one
for the non-shared degrees of, and one for the extended degrees of freedom. The
buffers are stored in a vector of vectors, one for each partition.

The non-shared degrees of freedom are the degrees of freedom that are owned
exclusively by a partition. Nevertheless, the partition needs also access to
degrees of freedom owned by other partitions.

(i) Multiplication of the stiffness matrix by a vector. The non-shared
partitioning is used. 

(ii) Preparation of the two-level preconditioner. The non-shared partitioning is
used. 

(iii) Solution of the systems of equations at the level of the partitions (level
1 of the two-level Schwarz preconditioner). The extended partitioning is used.

Here, the local degrees of freedom numbering is such that
`p.partition_list[i].entity_list[NONSHARED].local_receive_dofs[j]` are degrees
of freedom that partition `i` receives from partition `j`, and
`p.partition_list[j].entity_list[NONSHARED].local_send_dofs[i]` are degrees of
freedom that partition `j` sends to partition `i`.

`_lhs_update!` and `_rhs_update!` are functions that update the nonshared
degrees of freedom. The first is analogous to scatter, whereas the second is
analogous to gather. 

Scatter: owned degrees of freedom are known but the partition
needs to receive the values of the degrees of freedom that it doesn't own.

Gather: the partition needs to update the values of the degrees of freedom that
it owns, receiving contributions to those degrees of freedom from other
partitions.
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

using ..PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData, npartitions, NONSHARED, EXTENDED
using ..FinEtoolsDDMethods: set_up_timers, update_timer!, reset_timers!

import ..CGModule: vec_copyto!
import ..CGModule: vec_aypx!
import ..CGModule: vec_ypax!
import ..CGModule: vec_dot

"""
    DDCoNCMPIComm{PD<:CoNCPartitionData}

Communicator for MPI execution.
"""
struct DDCoNCMPIComm{MPIC, EL, PD<:CoNCPartitionData}
    comm::MPIC
    list_of_entity_lists::EL
    partition::PD
end

function DDCoNCMPIComm(comm, cpi, fes, make_matrix, make_interior_load)
    rank = MPI.Comm_rank(comm)
    partition = CoNCPartitionData(cpi, rank, fes, make_matrix, make_interior_load)
    return DDCoNCMPIComm(comm, cpi.list_of_entity_lists, partition)
end

struct PartitionedVector{DDC<:DDCoNCMPIComm, T<:Number}
    ddcomm::DDC
    buff_ns::Vector{T}
    buff_xt::Vector{T}
    temp_ns::Vector{Vector{T}}
end

function PartitionedVector(::Type{T}, ddcomm::DDC) where {T, DDC<:DDCoNCMPIComm}
    partition = ddcomm.partition
    buff_ns = fill(zero(T), length(partition.entity_list[NONSHARED].global_dofs))
    buff_xt = fill(zero(T), length(partition.entity_list[EXTENDED].global_dofs))
    Np = length(ddcomm.list_of_entity_lists)
    temp_ns = [fill(zero(T), length(partition.entity_list[NONSHARED].global_dofs))
                for i in 1:Np]
    return PartitionedVector(ddcomm, buff_ns, buff_xt, temp_ns)
end

"""
    vec_copyto!(a::PV, v::T) where {PV<:PartitionedVector, T}

Copy a single value into the partitioned vector.
"""
function vec_copyto!(a::PV, v::T) where {PV<:PartitionedVector, T}
    a.buff_ns .= v   
    a
end

"""
    vec_copyto!(a::PV, v::Vector{T}) where {PV<:PartitionedVector, T}

Copy a global vector into a partitioned vector.
"""
function vec_copyto!(a::PV, v::Vector{T}) where {PV<:PartitionedVector, T}
    partition = a.ddcomm.partition
    a.buff_ns .= v[partition.entity_list[NONSHARED].global_dofs]
    a
end

"""
    vec_copyto!(v::Vector{T}, a::PV) where {PV<:PartitionedVector, T}

Copy a partitioned vector into a global vector.
"""
function vec_copyto!(v::Vector{T}, a::PV) where {PV<:PartitionedVector, T}
    error("Implementation under review")
    partition = a.ddcomm.partition
    el = partition.entity_list
    ownd = el[NONSHARED].global_dofs[1:el[NONSHARED].num_own_dofs]
    lod = el[NONSHARED].global_to_local[ownd]
    v[ownd] .= a.buff_ns[lod]
    v
end

"""
    vec_copyto!(y::PV, x::PV) where {PV<:PartitionedVector}

Copy one partitioned vector to another.

The contents of `x` is copied into `y`.
"""
function vec_copyto!(y::PV, x::PV) where {PV<:PartitionedVector}
    @. y.buff_ns = x.buff_ns
    y
end

"""
    deepcopy(a::PV) where {PV<:PartitionedVector}

Create a deep copy of a partitioned vector.
"""
function deepcopy(a::PV) where {PV<:PartitionedVector}
    return PartitionedVector(a.ddcomm, deepcopy(a.buff_ns), deepcopy(a.buff_xt))
end

# Computes y = x + a y.
function vec_aypx!(y::PV, a, x::PV) where {PV<:PartitionedVector}
    @. y.buff_ns = a * y.buff_ns + x.buff_ns
    y
end

# Computes y += a x
function vec_ypax!(y::PV, a, x::PV) where {PV<:PartitionedVector}
    @. y.buff_ns = y.buff_ns + a * x.buff_ns
    y
end

function vec_dot(x::PV, y::PV) where {PV<:PartitionedVector}
    partition = y.ddcomm.partition
    own = 1:partition.entity_list[NONSHARED].num_own_dofs
    result = dot(y.buff_ns[own], x.buff_ns[own])
    return MPI.Allreduce(result, MPI.SUM, y.ddcomm.comm)
end

function _rhs_update!(p::PV) where {PV<:PartitionedVector}
    comm = p.ddcomm.comm
    loel = p.ddcomm.list_of_entity_lists
    i = p.ddcomm.partition.rank + 1
    # First I will send to other partitions
    local_send_dofs = loel[i].entity_list[NONSHARED].local_send_dofs
    for j in eachindex(local_send_dofs)
        if !isempty(local_send_dofs[j])
            req = MPI.Isend(p.buff_ns[local_send_dofs[j]], comm; dest = j)
        end
    end
    # Next I will receive from other partitions
    requests = MPI.Request[]
    local_receive_dofs = loel[i].entity_list[NONSHARED].local_receive_dofs
    for j in eachindex(local_receive_dofs)
        if !isempty(local_receive_dofs[j])
            push!(requests, MPI.Irecv!(p.buff_ns[local_receive_dofs[j]], comm; source = j))
        end
    end
    MPI.Waitall(requests)
end

function _lhs_update!(q::PV) where {PV<:PartitionedVector}
    comm = q.ddcomm.comm
    loel = q.ddcomm.list_of_entity_lists
    i = q.ddcomm.partition.rank + 1
    # First I will send to other partitions
    local_receive_dofs = loel[i].entity_list[NONSHARED].local_receive_dofs
    for j in eachindex(local_receive_dofs)
        if !isempty(local_receive_dofs[j])
            req = MPI.Isend(q.buff_ns[local_receive_dofs[j]], comm; dest = j)
        end
    end
    # Next I will receive from other partitions
    requests = MPI.Request[]
    local_send_dofs = loel[i].entity_list[NONSHARED].local_send_dofs
    for j in eachindex(local_send_dofs)
        if !isempty(local_send_dofs[j])
            q.temp_ns[j] .= 0
            push!(requests, MPI.Irecv!(q.temp_ns[j][local_send_dofs[j]], comm; source = j))
        end
    end
    MPI.Waitall(requests)
    q.buff_ns .= 0
    for j in eachindex(q.temp_ns)
        @. q.buff_ns += q.temp_ns
    end

    # partition_list = q.comm.partition_list
    # for i in eachindex(partition_list)
    #     local_send_dofs = partition_list[i].entity_list[NONSHARED].local_send_dofs
    #     for j in eachindex(local_send_dofs)
    #         if !isempty(local_send_dofs[j])
    #             local_receive_dofs = partition_list[j].entity_list[NONSHARED].local_receive_dofs
    #             q.buff_ns[i][local_send_dofs[j]] .+= q.buff_ns[j][local_receive_dofs[i]]
    #         end
    #     end
    # end
end

function aop!(q::PV, p::PV) where {PV<:PartitionedVector}
    _rhs_update!(p)
    q.buff_ns .= partition.Kns_ff * p.buff_ns
    _lhs_update!(q)
    q    
end

struct TwoLevelPreConditioner{DDC<:DDCoNCMPIComm{PD} where {PD<:CoNCPartitionData}, T, IT, FACTOR}
    ddcomm::DDC
    n::IT
    buff_Phi::SparseMatrixCSC{T, IT}
    Kr_ff_factor::FACTOR
    buffPp::Vector{T}
    buffKiPp::Vector{T}
end

function TwoLevelPreConditioner(ddcomm::DDC, Phi) where {DDC<:DDCoNCMPIComm{PD} where {PD<:CoNCPartitionData}}
    partition = ddcomm.partition
    n = size(Phi, 1)
    nr = size(Phi, 2)
    Kr_ff = spzeros(nr, nr)
    pel = partition.entity_list[NONSHARED]
    # First we work with all the degrees of freedom on the partition
    P = Phi[pel.global_dofs, :]
    Kr_ff += (P' * partition.Kns_ff * P)
    # the transformation matrices are now resized to only the own dofs
    ld = pel.local_own_dofs
    buff_Phi = P[ld, :]
    Kr_ff = Allreduce(Kr_ff, MPI.SUM, ddcomm.comm)
    Kr_ff_factor = lu(Kr_ff)
    buffPp = fill(zero(eltype(Kr_ff_factor)), nr)
    buffKiPp = fill(zero(eltype(Kr_ff_factor)), nr)
    return TwoLevelPreConditioner(ddcomm, n, buff_Phi, Kr_ff_factor, buffPp, buffKiPp)
end

function (pre::TwoLevelPreConditioner)(q::PV, p::PV) where {PV<:PartitionedVector}
    partition_list = p.comm.partition_list
    rhs_update_xt!(p)
    pre.buffPp .= zero(eltype(pre.buffPp))
    for i in eachindex(partition_list)
        ld = partition_list[i].entity_list[NONSHARED].local_own_dofs
        pre.buffPp .+= pre.buff_Phis[i]' * p.buff_ns[i][ld]
    end
    pre.buffKiPp .= pre.Kr_ff_factor \ pre.buffPp
    for i in eachindex(partition_list)
        q.buff_ns[i] .= 0
        ld = partition_list[i].entity_list[NONSHARED].local_own_dofs
        q.buff_ns[i][ld] .= pre.buff_Phis[i] * pre.buffKiPp
    end
    _lhs_update!(q)
    for i in eachindex(partition_list)
        q.buff_xt[i] .= partition_list[i].Kxt_ff_factor \ p.buff_xt[i]
    end
    lhs_update_xt!(q)
    q
end

function rhs_update_xt!(p::PV) where {PV<:PartitionedVector}
    partition_list = p.comm.partition_list
    for i in eachindex(partition_list)
        pin = partition_list[i].entity_list[EXTENDED]
        ld = pin.local_own_dofs
        p.buff_xt[i] .= 0
        p.buff_xt[i][ld] .= p.buff_ns[i][ld]
    end
    for i in eachindex(partition_list)
        local_receive_dofs = partition_list[i].entity_list[EXTENDED].local_receive_dofs
        for j in eachindex(local_receive_dofs)
            if !isempty(local_receive_dofs[j])
                local_send_dofs = partition_list[j].entity_list[EXTENDED].local_send_dofs
                p.buff_xt[i][local_receive_dofs[j]] .= p.buff_xt[j][local_send_dofs[i]]
            end
        end
    end
end

function lhs_update_xt!(q::PV) where {PV<:PartitionedVector}
    partition_list = q.comm.partition_list
    for i in eachindex(partition_list)
        local_send_dofs = partition_list[i].entity_list[EXTENDED].local_send_dofs
        for j in eachindex(local_send_dofs)
            if !isempty(local_send_dofs[j])
                local_receive_dofs = partition_list[j].entity_list[EXTENDED].local_receive_dofs
                q.buff_xt[i][local_send_dofs[j]] .+= q.buff_xt[j][local_receive_dofs[i]]
            end
        end
        qin = partition_list[i].entity_list[NONSHARED]
        ld = qin.local_own_dofs
        q.buff_ns[i][ld] .+= q.buff_xt[i][ld]
    end
end

function rhs(comm::C) where {C<:DDCoNCMPIComm} 
    rhs = deepcopy(comm.partition_list[1].rhs)
    rhs .= 0
    for i in eachindex(comm.partition_list)
        rhs .+= comm.partition_list[i].rhs
    end  
    return rhs
end

end # module DDCoNCMPIModule
