"""
    PartitionCoNCModule  

Module for operations on partitions of finite element models for solves based
on the Coherent Nodal Clusters.
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

# Node lists are constructed for the non-shared, and then by extending the
# partitions with the shared nodes. Connected elements are identified.
function _construct_entity_lists(fens, fes, n2e, overlap, node_1st_partitioning, dofnums, fr, i)
    nsnl = findall(x -> x == i, node_1st_partitioning)
    nocel = connectedelems(fes, nsnl, count(fens))
    bfes = meshboundary(subset(fes, nocel))
    cnl = connectednodes(bfes)
    # mark the members of the extended partition: start with the non-shared nodes
    ismember = fill(false, count(fens))
    ismember[nsnl] .= true 
    for ov in 1:overlap
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
    anl = findall(x -> x, ismember)
    ocel = connectedelems(fes, anl, count(fens))
    nonshared_dof_list, all_dof_list = _dof_lists(nsnl, anl, dofnums, fr)
    return (
        nonshared_nodes = nsnl,
        all_nodes = anl,
        connected_to_nonshared = nocel,
        connected_to_all = ocel,
        nonshared_dofs = nonshared_dof_list,
        all_dofs = all_dof_list
    )
end

"""
    _partition_entity_lists(fens, fes, Np, overlap, dofnums, fr)

Make entity lists for all grid subdomains.

The grid is partitioned into `Np` non-overlapping node partitions using the
Metis library. This means each partition owns it nodes, nodes are not shared
among partitions. 

The element partitions are extended by the given overlap. 
"""
function _partition_entity_lists(fens, fes, Np, overlap, dofnums, fr)
    femm = FEMMBase(IntegDomain(fes, PointRule()))
    C = connectionmatrix(femm, count(fens))
    g = Metis.graph(C; check_hermitian=true)
    node_1st_partitioning = Metis.partition(g, Np; alg=:KWAY)
    Np = maximum(node_1st_partitioning)
    n2e = FENodeToFEMap(fes, count(fens))
    entity_lists = []
    for i in 1:Np
        push!(entity_lists, _construct_entity_lists(fens, fes, n2e, overlap, node_1st_partitioning, dofnums, fr, i))
    end
    return entity_lists
end

"""
    subdomain_dof_lists(node_lists, dofnums, fr)

Collect the degree-of-freedom lists for all partitions.
"""
function _dof_lists(nonshared_node_list, all_node_list, dofnums, fr)
    nonshared_dof_list = Int[]
    sizehint!(nonshared_dof_list, prod(size(dofnums)))
    for n in nonshared_node_list
        for d in axes(dofnums, 2)
            if dofnums[n, d] in fr
                push!(nonshared_dof_list, dofnums[n, d]) # TO DO get rid of pushing
            end
        end
    end
    all_dof_list = Int[]
    sizehint!(all_dof_list, prod(size(dofnums)))
    for n in all_node_list
        for d in axes(dofnums, 2)
            if dofnums[n, d] in fr
                push!(all_dof_list, dofnums[n, d])
            end
        end
    end
    return nonshared_dof_list, all_dof_list
end

struct CoNCPartitioningInfo{NF<:NodalField{T, IT} where {T, IT}, EL} 
    u::NF
    entity_lists::EL
end

function nndof(cpi::CoNCPartitioningInfo, i)
    length(cpi.dof_lists[i].nonoverlapping)
end

function ondof(cpi::CoNCPartitioningInfo, i)
    length(cpi.dof_lists[i].overlapping)
end

function CoNCPartitioningInfo(fens, fes, Np, No, u::NodalField{T, IT}; visualize = false) where {T, IT}
    fr = dofrange(u, DOF_KIND_FREE)
    entity_lists = _partition_entity_lists(fens, fes, Np, No, u.dofnums, fr) # expensive
    if visualize
        p = 1
        for el in entity_lists
            vtkexportmesh("partition-$(p)-non-shared-elements.vtk", fens, subset(fes, el.connected_to_nonshared))
            sfes = FESetP1(reshape(el.nonshared_nodes, length(el.nonshared_nodes), 1))
            vtkexportmesh("partition-$(p)-non-shared-nodes.vtk", fens, sfes)
            vtkexportmesh("partition-$(p)-all-elements.vtk", fens, subset(fes, el.connected_to_all))
            sfes = FESetP1(reshape(el.all_nodes, length(el.all_nodes), 1))
            vtkexportmesh("partition-$(p)-all-nodes.vtk", fens, sfes)
            p += 1
        end
    end
    return CoNCPartitioningInfo(u, entity_lists)
end

function mean_partition_size(cpi::CoNCPartitioningInfo)
    return Int(round(mean([length(cpi.entity_lists[i].all_dof_list) for i in eachindex(cpi.entity_lists)])))
end

function npartitions(cpi::CoNCPartitioningInfo)
    return length(cpi.entity_lists)
end

mutable struct CoNCPartitionData{T, IT, FACTOR}
    rank::Int
    nonshared_K::SparseMatrixCSC{T, IT}
    all_K_factor::FACTOR
    rhs::Vector{T}
    nsdof::Vector{IT}
    ntempq::Vector{T}
    ntempp::Vector{T}
    alldof::Vector{IT}
    otempq::Vector{T}
    otempp::Vector{T}
end

function CoNCPartitionData(cpi::CPI) where {CPI<:CoNCPartitioningInfo}
    dummy = sparse([1],[1],[1.0],1,1)
    return CoNCPartitionData(
        0,
        spzeros(eltype(cpi.u.values), 0, 0),
        lu(dummy),
        zeros(eltype(cpi.u.values), 0),
        Int[],
        zeros(eltype(cpi.u.values), 0),
        zeros(eltype(cpi.u.values), 0),
        Int[],
        zeros(eltype(cpi.u.values), 0),
        zeros(eltype(cpi.u.values), 0)
    )
end

function CoNCPartitionData(cpi::CPI, 
    i, 
    fes, 
    make_matrix, 
    make_interior_load = nothing
    ) where {CPI<:CoNCPartitioningInfo}
    entity_lists = cpi.entity_lists
    fr = dofrange(cpi.u, DOF_KIND_FREE)
    dr = dofrange(cpi.u, DOF_KIND_DATA)
    # Compute the matrix for the non overlapping elements
    el = entity_lists[i].connected_to_nonshared
    Kn = make_matrix(subset(fes, el))
    # Now compute (contribution to) the reduced matrix for the global
    # preconditioner. At this point the matrix has the size of the overall
    # global stiffness matrix so that the reduced matrix Phi'*Kn_ff*Phi can be
    # computed later. It will need to be trimmed to only refer to the non
    # overlapping degrees of freedom.
    Kn_ff = Kn[fr, fr]
    # Compute the right hand side contribution
    u_d = gathersysvec(cpi.u, DOF_KIND_DATA)
    rhs = zeros(eltype(cpi.u.values), size(Kn_ff, 1))
    if norm(u_d, Inf) > 0
        rhs += - Kn[fr, dr] * u_d
    end
    if make_interior_load !== nothing
        rhs .+= make_interior_load(subset(fes, el))[fr]
    end
    Kn = nothing
    # Compute the matrix for the remaining (overlapping - nonoverlapping) elements
    el = setdiff(entity_lists[i].connected_to_all, entity_lists[i].connected_to_nonshared)
    Ke = make_matrix(subset(fes, el))
    Ka_ff = Kn_ff + Ke[fr, fr]
    Ke = nothing
    # Reduce the matrix to adjust the degrees of freedom referenced
    alldof = entity_lists[i].all_dofs
    Ka_ff = Ka_ff[alldof, alldof]
    # Reduce the matrix to adjust the degrees of freedom referenced
    nsdof = entity_lists[i].nonshared_dofs
    # Allocate some temporary vectors
    otempq = zeros(eltype(cpi.u.values), length(alldof))
    otempp = zeros(eltype(cpi.u.values), length(alldof))
    ntempq = zeros(eltype(cpi.u.values), length(nsdof))
    ntempp = zeros(eltype(cpi.u.values), length(nsdof))
    return CoNCPartitionData(i, Kn_ff, lu(Ka_ff), rhs, nsdof, ntempq, ntempp, alldof, otempq, otempp)
end

function partition_size(cpd::CoNCPartitionData)
    return length(cpd.alldof)
end

function rhs(partition_list)
    rhs = deepcopy(partition_list[1].rhs)
    for i in eachindex(partition_list)
        rhs .+= partition_list[i].rhs
    end  
    return rhs
end

end # module PartitionCoNCModule
