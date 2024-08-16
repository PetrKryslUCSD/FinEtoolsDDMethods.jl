"""
    PartitionCoNCDDSEQModule  

Module for operations on partitions of finite element models for solves based
on the Coherent Nodal Clusters.
"""
module PartitionCoNCDDSEQModule

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

struct CoNCPartitionMatrixCache{T, IT, FACTOR}
    nonoverlapping::SparseMatrixCSC{T, IT}
    overlapping_factor::FACTOR
    odof::Vector{IT}
    tempq::Vector{T}
    tempp::Vector{T}
end

struct CoNCPartitioning{T, IT, RFACTOR, MC} 
    Phi::SparseMatrixCSC{T, IT}
    Krfactor::RFACTOR
    caches::Vector{MC}
end

function CoNCPartitioning(fens, fes, nfpartitions, overlap, make_matrix, u::NodalField{T, IT}, Phi) where {T, IT}
    element_lists = subdomain_element_lists(fens, fes, nfpartitions, overlap)
    node_lists = subdomain_node_lists(element_lists, fes)
    fr = dofrange(u, DOF_KIND_FREE)
    dof_lists = subdomain_dof_lists(node_lists, u.dofnums, fr)
    Phi = Phi[fr, :]
    Kr_ff = spzeros(size(Phi, 2), size(Phi, 2))
    caches = CoNCPartitionMatrixCache[]
    for i in eachindex(element_lists)
        el = element_lists[i].nonoverlapping
        Kn = make_matrix(subset(fes, el))
        Kn_ff = Kn[fr, fr]
        Kr_ff .+= Phi' * Kn_ff * Phi
        el = setdiff(element_lists[i].all_connected, element_lists[i].nonoverlapping)
        Ke = make_matrix(subset(fes, el))
        Ko = Kn + Ke
        odof = dof_lists[i].overlapping
        Ko = Ko[odof, odof]
        tempq = zeros(T, length(odof))
        tempp = zeros(T, length(odof))
        push!(caches, 
            CoNCPartitionMatrixCache(Kn_ff, lu(Ko), odof, tempq, tempp)
        )
    end
    Krfactor = lu(Kr_ff)
    return CoNCPartitioning(Phi, Krfactor, caches)
end

function partition_multiply!(q, cp::CP, p) where {CP<:CoNCPartitioning}
    q .= zero(eltype(q))
    for i in eachindex(cp.caches)
        q .+= cp.caches[i].nonoverlapping * p
    end
    q
end

function precondition_solve!(q, cp::CP, p) where {CP<:CoNCPartitioning}
    q .= cp.Phi * (cp.Krfactor \ (cp.Phi' * p))
    for i in eachindex(cp.caches)
        d = cp.caches[i].odof
        cp.caches[i].tempp .= p[d]
        ldiv!(cp.caches[i].tempq, cp.caches[i].overlapping_factor, cp.caches[i].tempp)
        q[d] .+= cp.caches[i].tempq
        # q[d] .+= (cp.caches[i].overlapping_factor \ p[d])
    end
    q
end

end # module PartitionCoNCDDSEQModule
