"""
    PartitionCoNCModule  

Module for operations on partitions of finite element models for solves based on
the Coherent Nodal Clusters.

The grid is partitioned into `Np` non-overlapping node partitions using the
Metis library. This means each partition owns it nodes, nodes are not shared
among partitions. Neither are elements: elements are assigned uniquely to
partitions.

The partitions are extended by the given overlap using the connections among the
nodes. 

The communication is set up so that each partition has a list of lists of
degrees of freedom that it needs to receive from other partitions, one per
partition, and also list of lists of degrees of freedom that the partition needs
to send to other partitions (again one per partition).

The idea is that each partition is capable of performing part of an operation
required on the overall stiffness matrix. In particular, multiplication of the
stiffness matrix by a vector, construction of the global reduced matrix (all on
the owned partition mesh). Each partition also needs to support the local
preconditioning solve (on the extended partition mesh).
"""
module PartitionCoNCModule

__precompile__(true)

using FinEtools
using CoNCMOR
using SparseArrays
using Metis
using SparseArrays
using LinearAlgebra
import Base: size, eltype
import LinearAlgebra: mul!, eigen
using Statistics: mean
using ..FENodeToPartitionMapModule: FENodeToPartitionMap
using ..FinEtoolsDDMethods: allbytes
using ..FinEtoolsDDMethods: set_up_timers, update_timer!, reset_timers!
using ShellStructureTopo

# Node lists are constructed for the owned, and then by extending the
# partitions with the shared nodes. Connected elements are identified.
function _construct_node_lists_1(fens, fes, n2e, No, node_to_partition, i)
    # this is the owned node list
    nsnl = findall(x -> x == i, node_to_partition)
    nocel = connectedelems(fes, nsnl, count(fens))
    bfes = meshboundary(subset(fes, nocel))
    cnl = connectednodes(bfes)
    # mark the members of the extended partition: start with the owned nodes
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
        own_nodes = nsnl,
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
        push!(element_lists, (own_elements = uel, extended_elements = xtel))
    end # for i in eachindex(node_lists)
    return element_lists, element_to_partition
end

function _construct_communication_lists_own(node_lists, n2e, fes, node_to_partition, element_to_partition)
    Np = length(node_lists)
    comm_lists = [Int[] for i in 1:Np, j in 1:Np] # communication lists for each partition
    for p in eachindex(node_lists)
        l = node_lists[p]
        for n in l.own_nodes
            for e in n2e.map[n]
                if element_to_partition[e] == p # only if owned element
                    for on in fes.conn[e]
                        op = node_to_partition[on]
                        if op != p
                            push!(comm_lists[p, op], on)
                        end
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

struct EntityListsContainer{IT<:Integer}
    # All nodes that constitute the partition.
    nodes::Vector{IT}
    # All elements that constitute the partition.
    elements::Vector{IT}
    # Receive from nodes.
    receive_nodes::Vector{Vector{IT}}
    # Send to nodes.
    send_nodes::Vector{Vector{IT}}
    # All global dofs for the partition.
    global_dofs::Vector{IT}
    # Number of own dofs.
    num_own_dofs::IT
    # Local own dofs.
    ldofs_own_only::Vector{IT}
    # Local numbers of dofs received from other partitions.
    ldofs_other::Vector{Vector{IT}}
    # Local numbers of dofs sent to other partitions.
    ldofs_self::Vector{Vector{IT}}
    # Mapping from global to local dofs.
    dof_glob2loc::Vector{IT}
end

function _make_list_of_entity_lists(fens, fes, Np, No, dofnums, fr, node_to_partition = Int[])
    timers = set_up_timers("partition", "helper_lists", "list_of_entity_lists",)
    IT = eltype(fes.conn[1])
    if isempty(node_to_partition)
        t_ = time()
        femm = FEMMBase(IntegDomain(fes, PointRule()))
        C = connectionmatrix(femm, count(fens))
        @time g = Metis.graph(C; check_hermitian=true)
        node_to_partition = Metis.partition(g, Np; alg=:KWAY)
        update_timer!(timers, "partition", time() - t_)
    end
    Np = maximum(node_to_partition)
    t_ = time()
    n2e = FENodeToFEMap(fes, count(fens))
    node_lists =  _construct_node_lists(fens, fes, n2e, No, Np, node_to_partition)
    element_lists, element_to_partition = _construct_element_lists(fens, fes, n2e, node_lists, node_to_partition)
    own_comm = _construct_communication_lists_own(node_lists, n2e, fes, node_to_partition, element_to_partition)
    extended_comm = _construct_communication_lists_extended(node_lists, n2e, fes, node_to_partition)
    update_timer!(timers, "helper_lists", time() - t_)
    list_of_entity_lists = @NamedTuple{own::EntityListsContainer{IT}, extended::EntityListsContainer{IT}}[]
    t_ = time()
    for i in 1:Np
        # own
        nodes = node_lists[i].own_nodes
        elements = element_lists[i].own_elements
        receive_nodes_i = own_comm.receive_nodes[i]
        send_nodes_i = own_comm.send_nodes[i]
        global_dofs = _dof_list(node_lists[i].own_nodes, dofnums, fr)
        gdofs_own = deepcopy(global_dofs)
        num_own_dofs = length(gdofs_own)
        gdofs_other = [_dof_list(receive_nodes_i[j], dofnums, fr)
                        for j in eachindex(receive_nodes_i)]
        for j in eachindex(gdofs_other)
            global_dofs = vcat(global_dofs, gdofs_other[j])
        end
        dof_glob2loc = fill(0, prod(size(dofnums)))
        for j in eachindex(global_dofs)
            dof_glob2loc[global_dofs[j]] = j
        end
        ldofs_own_only = dof_glob2loc[gdofs_own]
        gdofs_self = [_dof_list(send_nodes_i[j], dofnums, fr)
                     for j in eachindex(send_nodes_i)]
        ldofs_other = [dof_glob2loc[gdofs_other[j]] for j in eachindex(gdofs_other)]
        ldofs_self = [dof_glob2loc[gdofs_self[j]] for j in eachindex(gdofs_self)]
        own = EntityListsContainer(
            nodes,
            elements,
            receive_nodes_i,
            send_nodes_i,
            global_dofs,
            num_own_dofs,
            ldofs_own_only,
            ldofs_other,
            ldofs_self,
            dof_glob2loc,
        )
        # extended
        nodes = node_lists[i].extended_nodes
        elements = element_lists[i].extended_elements
        receive_nodes_i = extended_comm.receive_nodes[i]
        send_nodes_i = extended_comm.send_nodes[i]
        global_dofs = _dof_list(node_lists[i].extended_nodes, dofnums, fr)
        num_own_dofs = own.num_own_dofs # the partitions own the same nodes
        gdofs_own = deepcopy(global_dofs[1:num_own_dofs])
        dof_glob2loc = fill(0, prod(size(dofnums)))
        for j in eachindex(global_dofs)
            dof_glob2loc[global_dofs[j]] = j
        end
        ldofs_own_only = dof_glob2loc[gdofs_own]
        gdofs_other = [_dof_list(receive_nodes_i[j], dofnums, fr)
                        for j in eachindex(receive_nodes_i)]
        gdofs_self = [_dof_list(send_nodes_i[j], dofnums, fr)
                            for j in eachindex(send_nodes_i)]
        ldofs_other = [dof_glob2loc[gdofs_other[j]] for j in eachindex(gdofs_other)]
        ldofs_self = [dof_glob2loc[gdofs_self[j]] for j in eachindex(gdofs_self)]
        extended = EntityListsContainer(
            nodes,
            elements,
            receive_nodes_i,
            send_nodes_i,
            global_dofs,
            num_own_dofs,
            ldofs_own_only,
            ldofs_other,
            ldofs_self,
            dof_glob2loc,
        )
        push!(list_of_entity_lists, (own = own, extended = extended))
    end
    update_timer!(timers, "list_of_entity_lists", time() - t_)
    @show timers
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

"""
    CoNCPartitioningInfo{NF<:NodalField{T, IT} where {T, IT}, EL} 

Partitioning info refers to the field (unknowns and data, for example
temperature or displacements), and to a list of entity lists: for each
partition, provide list of nodes, elements, degrees of freedom.
"""
struct CoNCPartitioningInfo{NF<:NodalField{T, IT} where {T, IT}, EL} 
    u::NF
    list_of_entity_lists::EL
end

function CoNCPartitioningInfo(fens, fes, Np, No, u::NodalField{T, IT}, node_to_partition) where {T, IT}
    fr = dofrange(u, DOF_KIND_FREE)
    @time list_of_entity_lists = _make_list_of_entity_lists(fens, fes, Np, No, u.dofnums, fr, node_to_partition)
    return CoNCPartitioningInfo(u, list_of_entity_lists)
end

function CoNCPartitioningInfo(fens, fes, Np, No, u::NodalField{T, IT}) where {T, IT}
    return CoNCPartitioningInfo(fens, fes, Np, No, u, Int[])
end

function mean_partition_size(cpi::CoNCPartitioningInfo)
    return Int(round(mean([length(el.extended.global_dofs) 
        for el in cpi.list_of_entity_lists])))
end

function npartitions(cpi::CoNCPartitioningInfo)
    return length(cpi.list_of_entity_lists)
end

"""
    CoNCPartitionData{EL, T, IT, FACTOR}

Data for a partition.
"""
mutable struct CoNCPartitionData{EL, T, IT, FACTOR}
    # Rank of the partition (its serial number, zero based).
    rank::IT
    # List of entities (nodes, elements, degrees of freedom).
    entity_list::EL
    # The stiffness matrix assembled from the owned elements.
    Kown_ff::SparseMatrixCSC{T, IT}
    # The factor (LU) of the stiffness matrix assembled from the extended elements.
    Kxt_ff_factor::FACTOR
    # A buffer for the righthand side vector.
    rhs::Vector{T}
end

function CoNCPartitionData(cpi::CPI, rank) where {CPI<:CoNCPartitioningInfo}
    dummy = sparse([1],[1],[1.0],1,1)
    return CoNCPartitionData(
        rank,
        cpi.list_of_entity_lists[rank+1],
        spzeros(eltype(cpi.u.values), 0, 0),
        lu(dummy),
        zeros(eltype(cpi.u.values), 0),
    )
end

"""
    CoNCPartitionData(cpi::CPI, 
    rank, 
    fes, 
    make_matrix, 
    make_interior_load = nothing
    ) where {CPI<:CoNCPartitioningInfo}

Create partition data.

# Arguments

- `cpi`: partitioning info
- `rank`: rank of the partition, zero based
- `fes`: finite element set 
- `make_matrix`: function that creates the matrix for a given finite element subset
- `make_interior_load`: function that creates the interior load vector for a given finite element subset

"""
function CoNCPartitionData(cpi::CPI, 
    rank, 
    fes, 
    make_matrix, 
    make_interior_load = nothing
    ) where {CPI<:CoNCPartitioningInfo}
    entity_list = cpi.list_of_entity_lists[rank+1]
    fr = dofrange(cpi.u, DOF_KIND_FREE)
    dr = dofrange(cpi.u, DOF_KIND_DATA)
    # Compute the matrix for the owned elements
    el = entity_list.own.elements
    Kown = make_matrix(subset(fes, el))
    # Trim to just the free degrees of freedom
    Kown_ff = Kown[fr, fr]
    # Compute the right hand side contribution
    u_d = gathersysvec(cpi.u, DOF_KIND_DATA)
    rhs = zeros(eltype(cpi.u.values), size(Kown_ff, 1))
    if norm(u_d, Inf) > 0
        rhs += - Kown[fr, dr] * u_d
    end
    if make_interior_load !== nothing
        rhs .+= make_interior_load(subset(fes, el))[fr]
    end
    # Compute the matrix for the remaining (overlapping - nonoverlapping) elements
    el = setdiff(entity_list.extended.elements, entity_list.own.elements)
    Kadd = make_matrix(subset(fes, el))
    Kxt_ff = Kown_ff + Kadd[fr, fr]
    Kadd = nothing
    # Reduce the matrix to adjust the degrees of freedom referenced
    d = entity_list.extended.global_dofs
    Kxt_ff = Kxt_ff[d, d]
    d = entity_list.own.global_dofs
    Kown_ff = Kown[d, d]
    Kown = nothing
    return CoNCPartitionData(rank, entity_list, Kown_ff, lu(Kxt_ff), rhs)
end

function partition_size(cpd::CoNCPartitionData)
    return length(cpd.entity_list.extended.global_dofs)
end

end # module PartitionCoNCModule
