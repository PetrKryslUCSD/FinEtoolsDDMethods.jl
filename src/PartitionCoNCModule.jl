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

function _construct_element_lists(fes, n2e, overlap, element_1st_partitioning, i)
    nonoverlapping_element_list = findall(x -> x == i, element_1st_partitioning)
    enl = deepcopy(nonoverlapping_element_list)
    touched = fill(false, count(fes)) 
    touched[enl] .= true 
    for ov in 1:overlap
        sfes = subset(fes, enl)
        bsfes = meshboundary(sfes)
        addenl = Int[]
        for e in eachindex(bsfes)
            for n in bsfes.conn[e]
                for ne in n2e.map[n]
                    if !touched[ne] 
                        touched[ne] = true
                        push!(addenl, ne)
                    end
                end
            end
        end
        enl = cat(enl, addenl; dims=1)
    end
    overlapping_element_list = enl
    n = connectednodes(subset(fes, overlapping_element_list))
    all_connected_element_list = connectedelems(fes, n, length(n2e.map))
    return (
        nonoverlapping=nonoverlapping_element_list,
        overlapping=overlapping_element_list,
        all_connected=all_connected_element_list
    )
end

"""
    subdomain_element_lists(fens, fes, npartitions, overlap)

Make element lists for all grid subdomains.

The grid is partitioned into `npartitions` non-overlapping element partitions using the
Metis library. The element partitions are extended by the given overlap. 
"""
function subdomain_element_lists(fens, fes, npartitions, overlap)
    femm = FEMMBase(IntegDomain(fes, PointRule()))
    C = dualconnectionmatrix(femm, fens, nodesperelem(boundaryfe(fes)))
    g = Metis.graph(C; check_hermitian=true)
    element_1st_partitioning = Metis.partition(g, npartitions; alg=:KWAY)
    npartitions = maximum(element_1st_partitioning)
    n2e = FENodeToFEMap(fes, count(fens))
    element_lists = []
    for i in 1:npartitions
        push!(element_lists, _construct_element_lists(fes, n2e, overlap, element_1st_partitioning, i))
    end
    return element_lists
end

"""
    subdomain_node_lists(element_lists, fes)

Make node lists for all grid subdomains.

List of named tuples of these nodelists is returned. 
"""
function subdomain_node_lists(element_lists, fes)
    node_lists = []
    for i in eachindex(element_lists)
        push!(node_lists,
            (
                nonoverlapping=connectednodes(subset(fes, element_lists[i].nonoverlapping)),
                overlapping=connectednodes(subset(fes, element_lists[i].overlapping))
            )
        )
    end
    return node_lists
end

"""
    subdomain_dof_lists(node_lists, dofnums, fr)

Collect the degree-of-freedom lists for all partitions.
"""
function subdomain_dof_lists(node_lists, dofnums, fr)
    dof_lists = []
    for n in node_lists
        nonoverlapping_dof_list = Int[]
        sizehint!(nonoverlapping_dof_list, prod(size(dofnums))) 
        for n in n.nonoverlapping
            for d in axes(dofnums, 2)
                if dofnums[n, d] in fr
                    push!(nonoverlapping_dof_list, dofnums[n, d]) # TO DO get rid of pushing
                end
            end
        end
        overlapping_dof_list = Int[]
        sizehint!(overlapping_dof_list, prod(size(dofnums))) 
        for n in n.overlapping
            for d in axes(dofnums, 2)
                if dofnums[n, d] in fr
                    push!(overlapping_dof_list, dofnums[n, d])
                end
            end
        end
        part = (overlapping=overlapping_dof_list,
                nonoverlapping=nonoverlapping_dof_list)
        push!(dof_lists, part)
    end
    return dof_lists
end

struct CoNCPartitioningInfo{NF<:NodalField{T, IT} where {T, IT}, EL, DL, BV} 
    u::NF
    element_lists::EL
    dof_lists::DL
    nbuffs::BV
    obuffs::BV
end

function CoNCPartitioningInfo(fens, fes, nfpartitions, overlap, u::NodalField{T, IT}) where {T, IT}
    element_lists = subdomain_element_lists(fens, fes, nfpartitions, overlap) # expensive
    node_lists = subdomain_node_lists(element_lists, fes) # cheap
    fr = dofrange(u, DOF_KIND_FREE)
    dof_lists = subdomain_dof_lists(node_lists, u.dofnums, fr) # intermediate
    nbuffs = [zeros(eltype(u.values), length(dof_lists[i].nonoverlapping)) for i in eachindex(dof_lists)]
    obuffs = [zeros(eltype(u.values), length(dof_lists[i].overlapping)) for i in eachindex(dof_lists)]
    return CoNCPartitioningInfo(u, element_lists, dof_lists, nbuffs, obuffs)
end

function mean_partition_size(cpi::CoNCPartitioningInfo)
    return Int(round(mean([length(cpi.dof_lists[i].overlapping) for i in eachindex(cpi.dof_lists)])))
end

function npartitions(cpi::CoNCPartitioningInfo)
    @assert length(cpi.element_lists) == length(cpi.dof_lists)
    return length(cpi.element_lists)
end

mutable struct CoNCPartitionData{T, IT, FACTOR}
    nonoverlapping_K::SparseMatrixCSC{T, IT}
    overlapping_K_factor::FACTOR
    rhs::Vector{T}
    ndof::Vector{IT}
    ntempq::Vector{T}
    ntempp::Vector{T}
    odof::Vector{IT}
    otempq::Vector{T}
    otempp::Vector{T}
end

function CoNCPartitionData(cpi::CPI) where {CPI<:CoNCPartitioningInfo}
    dummy = sparse([1],[1],[1.0],1,1)
    return CoNCPartitionData(
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
    element_lists = cpi.element_lists
    dof_lists = cpi.dof_lists
    fr = dofrange(cpi.u, DOF_KIND_FREE)
    dr = dofrange(cpi.u, DOF_KIND_DATA)
    # Compute the matrix for the non overlapping elements
    el = element_lists[i].nonoverlapping
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
    el = setdiff(element_lists[i].all_connected, element_lists[i].nonoverlapping)
    Ke = make_matrix(subset(fes, el))
    Ko_ff = Kn_ff + Ke[fr, fr]
    Ke = nothing
    # Reduce the matrix to adjust the degrees of freedom referenced
    odof = dof_lists[i].overlapping
    Ko_ff = Ko_ff[odof, odof]
    # Reduce the matrix to adjust the degrees of freedom referenced
    ndof = dof_lists[i].nonoverlapping
    # Allocate some temporary vectors
    otempq = zeros(eltype(cpi.u.values), length(odof))
    otempp = zeros(eltype(cpi.u.values), length(odof))
    ntempq = zeros(eltype(cpi.u.values), length(ndof))
    ntempp = zeros(eltype(cpi.u.values), length(ndof))
    return CoNCPartitionData(Kn_ff, lu(Ko_ff), rhs, ndof, ntempq, ntempp, odof, otempq, otempp)
end

function partition_size(cpd::CoNCPartitionData)
    return length(cpd.odof)
end

function rhs(partition_list)
    rhs = deepcopy(partition_list[1].rhs)
    for i in eachindex(partition_list)
        rhs .+= partition_list[i].rhs
    end  
    return rhs
end

end # module PartitionCoNCModule
