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
    # vtkexportmesh("i=$i" * "-enl" * "-final" * ".vtk", fens, subset(fes, enl))
    return (nonoverlapping = nonoverlapping_element_list, overlapping = overlapping_element_list)
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
    nodelists = []
    for i in eachindex(element_lists)
        push!(nodelists,
            (
                nonoverlapping=connectednodes(subset(fes, element_lists[i].nonoverlapping)),
                overlapping=connectednodes(subset(fes, element_lists[i].overlapping))
            )
        )
    end
    return nodelists
end

"""
    subdomain_dof_lists(nodelists, dofnums, fr)

Collect the degree-of-freedom lists for all partitions.
"""
function subdomain_dof_lists(nodelists, dofnums, fr)
    fpartitions = []
    for n in nodelists
        nonoverlapping_doflist = Int[]
        for n in n.nonoverlapping
            for d in axes(dofnums, 2)
                if dofnums[n, d] in fr
                    push!(nonoverlapping_doflist, dofnums[n, d])
                end
            end
        end
        overlapping_doflist = Int[]
        for n in n.overlapping
            for d in axes(dofnums, 2)
                if dofnums[n, d] in fr
                    push!(overlapping_doflist, dofnums[n, d])
                end
            end
        end
        part = (nodelists=n,
            doflists=(overlapping=overlapping_doflist,
                nonoverlapping=nonoverlapping_doflist)
        )
        push!(fpartitions, part)
    end
    return fpartitions
end

end # module PartitionCoNCDDMPIModule