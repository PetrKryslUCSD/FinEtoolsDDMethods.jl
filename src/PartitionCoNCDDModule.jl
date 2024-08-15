"""
    PartitionCoNCDDModule  

Module for operations on partitions of finite element models for solves based
on the Coherent Nodal Clusters.
"""
module PartitionCoNCDDModule

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

"""
    cluster_partitioning(fens, fes, element_labels, nelperpart)

Compute the coarse grid (cluster) partitioning.

The elements are partitioned into subsets of approximately equal size
(`nelperpart`). Then the nodes of the element subsets are assigned to clusters.

The element subsets are determined by the `element_labels` array: The elements
of a given label belong together.  Then the elements of the next subset are
partitioned,  and so on. The elements of label 0 are partitioned last.
"""
function cluster_partitioning(fens, fes, element_labels, nelperpart)
    max_el_label = maximum(element_labels)
    partitioning = zeros(Int, count(fens))
    nparts = 0
    for p in cat(1:max_el_label, 0, dims=1)
        el = findall(x -> x == p, element_labels)
        femm = FEMMBase(IntegDomain(subset(fes, el), PointRule()))
        C = dualconnectionmatrix(femm, fens, nodesperelem(boundaryfe(fes)))
        g = Metis.graph(C; check_hermitian=true)
        ndoms = Int(ceil(length(el) / nelperpart))
        subset_el_partitioning = Metis.partition(g, ndoms; alg=:KWAY)
        for e in eachindex(el)
            for n in fes.conn[el[e]]
                if partitioning[n] == 0
                    partitioning[n] = subset_el_partitioning[e] + nparts
                end
            end
        end
        nparts += maximum(subset_el_partitioning)
    end
    npartitions = maximum(partitioning)
    partitioning, npartitions
end


"""
    shell_cluster_partitioning(fens, fes, nelperpart = 50, crease_ang = 30/180*pi, cluster_max_normal_deviation = 2 * crease_ang)


Compute the coarse grid (cluster) partitioning of a shell structure.

The elements are partitioned into subsets with
`ShellStructureTopo.create_partitions()` based on the number of elements per
partition (`nelperpart`). Then the nodes of the element subsets are assigned to
clusters.
"""
function shell_cluster_partitioning(fens, fes, nelperpart = 50, crease_ang = 30/180*pi, cluster_max_normal_deviation = 2 * crease_ang)
    partitioning = zeros(Int, count(fens))
    surfids, partitionids, surface_nelperpart = ShellStructureTopo.create_partitions(fens, fes, nelperpart;
        crease_ang=crease_ang, cluster_max_normal_deviation=cluster_max_normal_deviation)
    for i in eachindex(fes)
        for k in fes.conn[i]
            partitioning[k] = partitionids[i]
        end
    end
    npartitions = maximum(partitioning)
    partitioning, npartitions
end

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

function patch_coordinates(panelX, list)
    krondelta(i, k) = i == k ? 1.0 : 0.0
    lX = panelX[list, :]
    center = mean(lX, dims=1)
    for j in axes(lX, 1)
        lX[j, :] -= center[:]
    end
    It = fill(0.0, 3, 3)
    for j in axes(lX, 1)
        r2 = dot(lX[j, :], lX[j, :])
        for i in 1:3
            for k in i:3
                It[i, k] += krondelta(i, k) * r2 - lX[j, i] * lX[j, k]
            end
        end
    end
    for i in 1:3
        for k in 1:i-1
            It[i, k] = It[k, i]
        end
    end
    @assert It == It'
    epsol = eigen(It)
    normal = epsol.vectors[:, 3]
    e1 = epsol.vectors[:, 1]
    e2 = epsol.vectors[:, 2]
    lxy = fill(0.0, length(list), 2)
    for j in axes(lX, 1)
        lxy[j, 1] = dot(lX[j, :], e1)
        lxy[j, 2] = dot(lX[j, :], e2)
    end
    return lxy
end

function fine_grid_partitions(fens, fes, nfpartitions, overlap, dofnums, fr)
    nodelists = fine_grid_node_lists(fens, fes, nfpartitions, overlap)
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

end # module PartitionCoNCDDModule
