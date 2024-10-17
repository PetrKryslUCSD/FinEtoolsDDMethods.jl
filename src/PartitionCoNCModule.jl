"""
    PartitionCoNCModule  

Module for operations on partitions of finite element models for solves based on
the Coherent Nodal Clusters.

The grid is partitioned into `Np` non-overlapping node partitions using the
Metis library. This means each partition owns it nodes, nodes are not shared
among partitions. 

The partitions are extended by the given overlap using the connections among the
nodes. 
"""
module PartitionCoNCModule

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
using ..FinEtoolsDDMethods: allbytes
using ShellStructureTopo

const NONSHARED = 1
const EXTENDED = 2

# Node lists are constructed for the non-shared, and then by extending the
# partitions with the shared nodes. Connected elements are identified.
function _construct_node_lists_1(fens, fes, n2e, No, node_to_partition, i)
    # this is the non-shared node list
    nsnl = findall(x -> x == i, node_to_partition)
    nocel = connectedelems(fes, nsnl, count(fens))
    bfes = meshboundary(subset(fes, nocel))
    cnl = connectednodes(bfes)
    # mark the members of the extended partition: start with the non-shared nodes
    ismember = fill(false, count(fens))
    ismember[nsnl] .= true 
    for _ in 1:No
        addednl = Int[]
        sizehint!(addednl, length(cnl))
        for cn in cnl
            for e in n2e.map[cn] # neighbor
                for on in fes.conn[e] # node
                    if !ismember[on]
                        ismember[on] = true
                        push!(addednl, on)
                    end
                end
            end
        end
        cnl = addednl
    end
    # the all node list consists of all the nodes reached within a given number of overlaps
    xtnl = findall(x -> x, ismember)
    return (
        nonshared_nodes = nsnl,
        extended_nodes = vcat(nsnl, setdiff(xtnl, nsnl)),
    )
end

function _construct_node_lists(fens, fes, n2e, No, Np, node_to_partition)
    node_lists = []
    for i in 1:Np
        push!(node_lists, _construct_node_lists_1(fens, fes, n2e, No, node_to_partition, i))
    end
    return node_lists
end

function _construct_element_lists(fens, fes, n2e, node_lists, node_to_partition)
    Np = length(node_lists)
    # Record assignment of the elements to partitions. This assignment is
    # unique: each element is assigned to the partition whose nodes reference it
    # the most. Ties are broken.
    element_to_partition = fill(0, count(fes))
    nodebuf = fill(0, nodesperelem(fes))
    numpartitions = fill(0, Np)
    ix = fill(0, Np)
    for n in eachindex(n2e.map)
        for e in n2e.map[n]
            if element_to_partition[e] == 0
                for k in eachindex(nodebuf)
                    nodebuf[k] = node_to_partition[fes.conn[e][k]]
                end
                numpartitions .= 0
                for i in eachindex(nodebuf)
                    numpartitions[nodebuf[i]] += 1
                end
                ix = sortperm!(ix, numpartitions)
                if length(ix) > 1  &&  numpartitions[ix[end]] > numpartitions[ix[end-1]]
                    element_to_partition[e] = ix[end] # assign to the most connected partition
                else # if there is a tie, assign to the partition with the smallest number
                    # numpartitions =  [9, 1, 8, 10, 10, 3, 0, 10]
                    # ix = sortperm!(ix, numpartitions)
                    k = 1
                    for j in length(ix):-1:1
                        # @show j, numpartitions[ix[j]], numpartitions[ix[end]]
                        if numpartitions[ix[j]] < numpartitions[ix[end]]
                            k = j + 1
                            break
                        end
                    end
                    # @show k, ix[k:end], minimum(ix[k:end])
                    element_to_partition[e] = minimum(ix[k:end])
                end
            end
        end
    end
    # Now find all elements that support the extended partition nodes
    element_lists = []
    for i in eachindex(node_lists)
        xtel = connectedelems(fes, node_lists[i].extended_nodes, count(fens))
        uel = findall(x -> x == i, element_to_partition)
        push!(element_lists, (nonshared_elements = uel, extended_elements = xtel))
    end # for i in eachindex(node_lists)
    return element_lists
end

function _construct_communication_lists_nonshared(node_lists, n2e, fes, node_to_partition)
    Np = length(node_lists)
    comm_lists = [Int[] for i in 1:Np, j in 1:Np] # communication lists for each partition
    for p in eachindex(node_lists)
        l = node_lists[p]
        for n in l.nonshared_nodes
            for e in n2e.map[n]
                for on in fes.conn[e]
                    op = node_to_partition[on]
                    if op != p
                        push!(comm_lists[p, op], on)
                    end
                end
            end
        end
    end
    for i in 1:Np
        for j in 1:Np
            comm_lists[i, j] = unique(comm_lists[i, j])
        end
    end
    # comm_lists[p, op] is the list of nodes whose data partition p needs from partition op
    receive_nodes = [comm_lists[i, :]  for i in 1:Np]
    send_nodes = [comm_lists[:, i]  for i in 1:Np]
    return (receive_nodes = receive_nodes, send_nodes = send_nodes)
end

function _construct_communication_lists_extended(node_lists, n2e, fes, node_to_partition)
    Np = length(node_lists)
    comm_lists = [Int[] for i in 1:Np, j in 1:Np] # communication lists for each partition
    for p in eachindex(node_lists)
        l = node_lists[p]
        for n in l.extended_nodes
            op = node_to_partition[n]
            if op != p
                push!(comm_lists[p, op], n)
            end
        end
    end
    for i in 1:Np
        for j in 1:Np
            comm_lists[i, j] = unique(comm_lists[i, j])
        end
    end
    # comm_lists[p, op] is the list of nodes that partition p needs from partition op
    receive_nodes = [comm_lists[i, :]  for i in 1:Np]
    send_nodes = [comm_lists[:, i]  for i in 1:Np]
    return (receive_nodes = receive_nodes, send_nodes = send_nodes)
end

struct EntityListNonShared{IT<:Integer}
    # All nodes that constitute the partition.
    nodes::Vector{IT}
    # All elements that constitute the partition.
    elements::Vector{IT}
    # Nodes for which dofs are received from other partitions.
    receive_nodes::Vector{Vector{IT}}
    # Nodes for which dofs are sent to other partitions.
    send_nodes::Vector{Vector{IT}}
    # All global dofs for the partition.
    global_dofs::Vector{IT}
    # Number of own dofs.
    num_own_dofs::IT
    # Global own dofs.
    global_own_dofs::Vector{IT}
    # Local own dofs.
    local_own_dofs::Vector{IT}
    # Global dofs received from other partitions.
    global_receive_dofs::Vector{Vector{IT}}
    # Global dofs sent to other partitions.
    global_send_dofs::Vector{Vector{IT}}
    # Local numbers of dofs received from other partitions.
    local_receive_dofs::Vector{Vector{IT}}
    # Local numbers of dofs sent to other partitions.
    local_send_dofs::Vector{Vector{IT}}
    # Mapping from global to local dofs.
    global_to_local::Vector{IT}
end

struct EntityListExtended{IT<:Integer}
    # All nodes that constitute the partition.
    nodes::Vector{IT}
    # All elements that constitute the partition.
    elements::Vector{IT}
    # Nodes for which dofs are received from other partitions.
    receive_nodes::Vector{Vector{IT}}
    # Nodes for which dofs are sent to other partitions.
    send_nodes::Vector{Vector{IT}}
    # All global dofs for the partition.
    global_dofs::Vector{IT}
    # Number of own dofs.
    num_own_dofs::IT
    # Global own dofs.
    global_own_dofs::Vector{IT}
    # Local own dofs.
    local_own_dofs::Vector{IT}
    # Global dofs received from other partitions.
    global_receive_dofs::Vector{Vector{IT}}
    # Global dofs sent to other partitions.
    global_send_dofs::Vector{Vector{IT}}
    # Local numbers of dofs received from other partitions.
    local_receive_dofs::Vector{Vector{IT}}
    # Local numbers of dofs sent to other partitions.
    local_send_dofs::Vector{Vector{IT}}
    # Mapping from global to local dofs.
    global_to_local::Vector{IT}
end

function _make_list_of_entity_lists(fens, fes, Np, No, dofnums, fr)
    femm = FEMMBase(IntegDomain(fes, PointRule()))
    C = connectionmatrix(femm, count(fens))
    g = Metis.graph(C; check_hermitian=true)
    node_to_partition = Metis.partition(g, Np; alg=:KWAY)
    Np = maximum(node_to_partition)
    n2e = FENodeToFEMap(fes, count(fens))
    node_lists =  _construct_node_lists(fens, fes, n2e, No, Np, node_to_partition)
    nonshared_comm = _construct_communication_lists_nonshared(node_lists, n2e, fes, node_to_partition)
    extended_comm = _construct_communication_lists_extended(node_lists, n2e, fes, node_to_partition)
    element_lists = _construct_element_lists(fens, fes, n2e, node_lists, node_to_partition)
    list_of_entity_lists = []
    for i in 1:Np
        # nonshared
        nodes = node_lists[i].nonshared_nodes
        elements = element_lists[i].nonshared_elements
        receive_nodes = nonshared_comm.receive_nodes[i]
        send_nodes = nonshared_comm.send_nodes[i]
        global_dofs = _dof_list(node_lists[i].nonshared_nodes, dofnums, fr)
        global_own_dofs = deepcopy(global_dofs)
        num_own_dofs = length(global_own_dofs)
        global_receive_dofs = [_dof_list(nonshared_comm.receive_nodes[i][j], dofnums, fr)
                        for j in eachindex(nonshared_comm.receive_nodes[i])]
        for j in eachindex(global_receive_dofs)
            global_dofs = vcat(global_dofs, global_receive_dofs[j])
        end
        global_to_local = fill(0, prod(size(dofnums)))
        for j in eachindex(global_dofs)
            global_to_local[global_dofs[j]] = j
        end
        local_own_dofs = global_to_local[global_own_dofs]
        global_send_dofs = [_dof_list(nonshared_comm.send_nodes[i][j], dofnums, fr)
                     for j in eachindex(nonshared_comm.send_nodes[i])]
        local_receive_dofs = [global_to_local[global_receive_dofs[j]] for j in eachindex(global_receive_dofs)]
        local_send_dofs = [global_to_local[global_send_dofs[j]] for j in eachindex(global_send_dofs)]
        nonshared = EntityListNonShared(
            nodes,
            elements,
            receive_nodes,
            send_nodes,
            global_dofs,
            num_own_dofs,
            global_own_dofs,
            local_own_dofs,
            global_receive_dofs,
            global_send_dofs,
            local_receive_dofs,
            local_send_dofs,
            global_to_local,
        )
        # extended
        nodes = node_lists[i].extended_nodes
        elements = element_lists[i].extended_elements
        receive_nodes = extended_comm.receive_nodes[i]
        send_nodes = extended_comm.send_nodes[i]
        global_dofs = _dof_list(node_lists[i].extended_nodes, dofnums, fr)
        num_own_dofs = nonshared.num_own_dofs # the partitions own the same nodes
        global_own_dofs = deepcopy(global_dofs[1:num_own_dofs])
        global_receive_dofs = [_dof_list(extended_comm.receive_nodes[i][j], dofnums, fr)
                        for j in eachindex(extended_comm.receive_nodes[i])]
        global_to_local = fill(0, prod(size(dofnums)))
        for j in eachindex(global_dofs)
            global_to_local[global_dofs[j]] = j
        end
        local_own_dofs = global_to_local[global_own_dofs]
        global_send_dofs = [_dof_list(extended_comm.send_nodes[i][j], dofnums, fr)
                            for j in eachindex(extended_comm.send_nodes[i])]
        local_receive_dofs = [global_to_local[global_receive_dofs[j]] for j in eachindex(global_receive_dofs)]
        local_send_dofs = [global_to_local[global_send_dofs[j]] for j in eachindex(global_send_dofs)]
        extended = EntityListExtended(
            nodes,
            elements,
            receive_nodes,
            send_nodes,
            global_dofs,
            num_own_dofs,
            global_own_dofs,
            local_own_dofs,
            global_receive_dofs,
            global_send_dofs,
            local_receive_dofs,
            local_send_dofs,
            global_to_local,
        )
        push!(list_of_entity_lists, [nonshared, extended])
    end
    return list_of_entity_lists
end

function _dof_list(node_list, dofnums, fr)
    dof_list = Int[]
    sizehint!(dof_list, length(node_list) * size(dofnums, 2))
    for n in node_list
        for d in axes(dofnums, 2)
            if dofnums[n, d] in fr
                push!(dof_list, dofnums[n, d]) # TO DO get rid of pushing
            end
        end
    end
    return dof_list
end

struct CoNCPartitioningInfo{NF<:NodalField{T, IT} where {T, IT}, EL} 
    u::NF
    list_of_entity_lists::EL
end

function CoNCPartitioningInfo(fens, fes, Np, No, u::NodalField{T, IT}) where {T, IT}
    fr = dofrange(u, DOF_KIND_FREE)
    list_of_entity_lists = _make_list_of_entity_lists(fens, fes, Np, No, u.dofnums, fr) # expensive
    return CoNCPartitioningInfo(u, list_of_entity_lists)
end

function mean_partition_size(cpi::CoNCPartitioningInfo)
    return Int(round(mean([length(cpi.entity_lists[i].all_dof_list) for i in eachindex(cpi.entity_lists)])))
end

function npartitions(cpi::CoNCPartitioningInfo)
    return length(cpi.list_of_entity_lists)
end

mutable struct CoNCPartitionData{EL, T, IT, FACTOR}
    rank::IT
    entity_list::EL
    Kns_ff::SparseMatrixCSC{T, IT}
    Kxt_ff_factor::FACTOR
    rhs::Vector{T}
end

function CoNCPartitionData(cpi::CPI, rank) where {CPI<:CoNCPartitioningInfo}
    dummy = sparse([1],[1],[1.0],1,1)
    return CoNCPartitionData(
        rank,
        cpi.list_of_entity_lists[rank],
        spzeros(eltype(cpi.u.values), 0, 0),
        lu(dummy),
        zeros(eltype(cpi.u.values), 0),
    )
end

function CoNCPartitionData(cpi::CPI, 
    rank, 
    fes, 
    make_matrix, 
    make_interior_load = nothing
    ) where {CPI<:CoNCPartitioningInfo}
    entity_list = cpi.list_of_entity_lists[rank]
    fr = dofrange(cpi.u, DOF_KIND_FREE)
    dr = dofrange(cpi.u, DOF_KIND_DATA)
    # Compute the matrix for the non shared elements
    el = entity_list[NONSHARED].elements
    Kns = make_matrix(subset(fes, el))
    # Trim to just the free degrees of freedom
    Kns_ff = Kns[fr, fr]
    # Compute the right hand side contribution
    u_d = gathersysvec(cpi.u, DOF_KIND_DATA)
    rhs = zeros(eltype(cpi.u.values), size(Kns_ff, 1))
    if norm(u_d, Inf) > 0
        rhs += - Kns[fr, dr] * u_d
    end
    if make_interior_load !== nothing
        rhs .+= make_interior_load(subset(fes, el))[fr]
    end
    # Compute the matrix for the remaining (overlapping - nonoverlapping) elements
    el = setdiff(entity_list[EXTENDED].elements, entity_list[NONSHARED].elements)
    Kadd = make_matrix(subset(fes, el))
    Kxt_ff = Kns_ff + Kadd[fr, fr]
    Kadd = nothing
    # Reduce the matrix to adjust the degrees of freedom referenced
    d = entity_list[EXTENDED].global_dofs
    Kxt_ff = Kxt_ff[d, d]
    d = entity_list[NONSHARED].global_dofs
    Kns_ff = Kns[d, d]
    Kns = nothing
    return CoNCPartitionData(rank, entity_list, Kns_ff, lu(Kxt_ff), rhs)
end

function partition_size(cpd::CoNCPartitionData)
    return length(cpd.entity_list[EXTENDED].global_dofs)
end

function rhs(partition_list)
    rhs = deepcopy(partition_list[1].rhs)
    for i in eachindex(partition_list)
        rhs .+= partition_list[i].rhs
    end  
    return rhs
end

end # module PartitionCoNCModule
