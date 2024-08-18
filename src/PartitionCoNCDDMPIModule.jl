"""
    PartitionCoNCDDMPIModule  

Module for operations on partitions of finite element models for solves based
on the Coherent Nodal Clusters.
"""
module PartitionCoNCDDMPIModule

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
        for n in n.nonoverlapping
            for d in axes(dofnums, 2)
                if dofnums[n, d] in fr
                    push!(nonoverlapping_dof_list, dofnums[n, d])
                end
            end
        end
        overlapping_dof_list = Int[]
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

struct CoNCPartitioningInfo{NF<:NodalField{T, IT} where {T, IT}, EL, DL} 
    u::NF
    element_lists::EL
    dof_lists::DL
end

function CoNCPartitioningInfo(fens, fes, nfpartitions, overlap, u::NodalField{T, IT}) where {T, IT}
    element_lists = subdomain_element_lists(fens, fes, nfpartitions, overlap)
    node_lists = subdomain_node_lists(element_lists, fes)
    fr = dofrange(u, DOF_KIND_FREE)
    dof_lists = subdomain_dof_lists(node_lists, u.dofnums, fr)
    return CoNCPartitioningInfo(u, element_lists, dof_lists)
end

struct CoNCPartitionData{T, IT, FACTOR}
    nonoverlapping_K::SparseMatrixCSC{T, IT}
    reduced_K::SparseMatrixCSC{T, IT}
    overlapping_K_factor::FACTOR
    ndof::Vector{IT}
    ntempq::Vector{T}
    ntempp::Vector{T}
    odof::Vector{IT}
    otempq::Vector{T}
    otempp::Vector{T}
end

function CoNCPartitionData(cpi::CPI, i, fes, Phi, make_matrix) where {CPI<:CoNCPartitioningInfo}
    element_lists = cpi.element_lists
    dof_lists = cpi.dof_lists
    fr = dofrange(cpi.u, DOF_KIND_FREE)
    Phi = Phi[fr, :]
    Kr_ff = spzeros(size(Phi, 2), size(Phi, 2))
    el = element_lists[i].nonoverlapping
    Kn = make_matrix(subset(fes, el))
    Kn_ff = Kn[fr, fr]
    Kr_ff .+= Phi' * Kn_ff * Phi
    el = setdiff(element_lists[i].all_connected, element_lists[i].nonoverlapping)
    Ke = make_matrix(subset(fes, el))
    Ko = Kn + Ke
    odof = dof_lists[i].overlapping
    Ko = Ko[odof, odof]
    otempq = zeros(eltype(cpi.u.values), length(odof))
    otempp = zeros(eltype(cpi.u.values), length(odof))
    ndof = dof_lists[i].nonoverlapping
    ntempq = zeros(eltype(cpi.u.values), length(ndof))
    ntempp = zeros(eltype(cpi.u.values), length(ndof))
    Kn_ff = Kn_ff[ndof, ndof]
    return CoNCPartitionData(Kn_ff, Kr_ff, lu(Ko), ndof, ntempq, ntempp, odof, otempq, otempp)
end

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

end # module PartitionCoNCDDMPIModule
