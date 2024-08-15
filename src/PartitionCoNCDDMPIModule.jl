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


function _constructnodelists(fes, n2e, overlap, element_1st_partitioning, i)
    enl = findall(x -> x == i, element_1st_partitioning)
    nonoverlapping_nodelist = connectednodes(subset(fes, enl))
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
    return (nonoverlapping = nonoverlapping_nodelist, overlapping = connectednodes(subset(fes, enl)))
end

"""
    subdomain_node_lists(fens, fes, npartitions, overlap)

Make node lists for all partitions of the grid subdomains.

The grid is partitioned into `npartitions` using the Metis library. The
partitions are extended by the given overlap. Then for each extended partition
of elements, the list of nodes is collected.

List of named tuples of these nodelists is returned. 
"""
function subdomain_node_lists(fens, fes, npartitions, overlap)
    femm = FEMMBase(IntegDomain(fes, PointRule()))
    C = dualconnectionmatrix(femm, fens, nodesperelem(boundaryfe(fes)))
    g = Metis.graph(C; check_hermitian=true)
    element_1st_partitioning = Metis.partition(g, npartitions; alg=:KWAY)
    npartitions = maximum(element_1st_partitioning)
    n2e = FENodeToFEMap(fes, count(fens))
    nodelists = []
    for i in 1:npartitions
        push!(nodelists, _constructnodelists(fes, n2e, overlap, element_1st_partitioning, i))
    end
    nodelists
end

function fine_grid_partitions(fens, fes, nfpartitions, overlap, dofnums, fr)
    nodelists = subdomain_node_lists(fens, fes, nfpartitions, overlap)
    @assert length(nodelists) == nfpartitions
    fpartitions = []
    for nodelist in nodelists
        doflist = Int[]
        for n in nodelist
            for d in axes(dofnums, 2)
                if dofnums[n, d] in fr
                    push!(doflist, dofnums[n, d])
                end
            end
        end
        part = (nodelist = nodelist, doflist = doflist)
        push!(fpartitions, part)
    end
    fpartitions
end

function preconditioner(fpartitions, Phi, K)
    PhiT = transpose(Phi)
    Kr = PhiT * K * Phi
    Krfactor = lu(Kr)
    __partitions = []
    for part in fpartitions
        pK = K[part.doflist, part.doflist]
        pKfactor = lu(pK)
        __part = (factor = pKfactor, doflist = part.doflist)
        push!(__partitions, __part)
    end
    function M!(q, p)
        q .= Phi * (Krfactor \ (PhiT * p))
        for part in __partitions
            q[part.doflist] .+= (part.factor \ p[part.doflist])
        end
        q
    end
    M!
end

end # module PartitionCoNCDDMPIModule
