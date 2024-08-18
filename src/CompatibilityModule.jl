"""
    CompatibilityModule  

Module for compatibility with the old version that went into the review.
"""
module CompatibilityModule

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
using DataDrop

function _make_overlapping_partition(fes, n2e, overlap, element_1st_partitioning, i)
    enl = findall(x -> x == i, element_1st_partitioning)
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
    # vtkexportmesh("i=$i" * "-enl" * "-final" * ".vtk", fens, subset(fes, enl))
    return connectednodes(subset(fes, enl))
end

"""
    fine_grid_node_lists(fens, fes, npartitions, overlap)

Make node lists for all partitions of the fine grid.

The fine grid is partitioned into `npartitions` using the Metis library. The
partitions are extended by the given overlap. Then for each extended partition
of elements, the list of nodes is collected.

List of these node-lists is returned.
"""
function fine_grid_node_lists(fens, fes, npartitions, overlap)
    femm = FEMMBase(IntegDomain(fes, PointRule()))
    C = dualconnectionmatrix(femm, fens, nodesperelem(boundaryfe(fes)))
    g = Metis.graph(C; check_hermitian=true)
    element_1st_partitioning = Metis.partition(g, npartitions; alg=:KWAY)
    npartitions = maximum(element_1st_partitioning)
    n2e = FENodeToFEMap(fes, count(fens))
    nodelists = []
    for i in 1:npartitions
        push!(nodelists, _make_overlapping_partition(fes, n2e, overlap, element_1st_partitioning, i))
    end
    nodelists
end

end # module CompatibilityModule
