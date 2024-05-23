module mbas001
using Test
using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsDDParallel
using Metis
function test()
    ndoms = 3
    fens, fes = T3block(1.0, 1.0, 10, 13)
    femm1 = FEMMBase(IntegDomain(fes, SimplexRule(3, 1)))
    C = dualconnectionmatrix(femm1, fens, nodesperelem(boundaryfe(fes)))
    g = Metis.graph(C; check_hermitian=true)
    element_partitioning = Metis.partition(g, ndoms; alg=:KWAY)
    VTK.vtkexportmesh("mbas001-element_partitioning.vtk", connasarray(fes), fens.xyz, VTK.T3; scalars=[("element_partitioning", element_partitioning)])
    n2p = FENodeToPartitionMap(fens, fes, element_partitioning)
    inodes = []
    for i in eachindex(n2p.map)
        if length(n2p.map[i]) > 1
            push!(inodes, i)
        end
    end
    ifes = FESetP1(inodes)
    VTK.vtkexportmesh("mbas001-interfaces.vtk", connasarray(ifes), fens.xyz, VTK.P1;)
    true
end
test()
nothing
end

module mbas002
using Test
using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsDDParallel
using Metis
function make_partition_mesh(fens, fes, n2p, element_partitioning, partition)
    pfes = subset(fes, findall(y -> y == partition, element_partitioning))
    connected = findunconnnodes(fens, pfes)
    fens, new_numbering = compactnodes(fens, connected)
    pfes = renumberconn!(pfes, new_numbering)
    global_node_numbers = [i for i in eachindex(new_numbering) 
        if new_numbering[i] != 0]
    return fens, pfes, global_node_numbers
end
function test()
    ndoms = 3
    fens, fes = T3block(1.0, 1.0, 7, 9)
    femm1 = FEMMBase(IntegDomain(fes, SimplexRule(3, 1)))
    C = dualconnectionmatrix(femm1, fens, nodesperelem(boundaryfe(fes)))
    g = Metis.graph(C; check_hermitian=true)
    element_partitioning = Metis.partition(g, ndoms; alg=:KWAY)
    VTK.vtkexportmesh("mbas002-element_partitioning.vtk", connasarray(fes), fens.xyz, VTK.T3; scalars=[("element_partitioning", element_partitioning)])
    n2p = FENodeToPartitionMap(fens, fes, element_partitioning)
    inodes = []
    for i in eachindex(n2p.map)
        if length(n2p.map[i]) > 1
            push!(inodes, i)
        end
    end
    ifes = FESetP1(inodes)
    VTK.vtkexportmesh("mbas002-interfaces.vtk", connasarray(ifes), fens.xyz, VTK.P1;)

    for partition in unique(element_partitioning)
        fens1, fes1, global_node_numbers1 = make_partition_mesh(fens, fes, n2p, element_partitioning, partition)
        VTK.vtkexportmesh("mbas002-partition-$(partition).vtk", connasarray(fes1), fens1.xyz, VTK.T3)
        local_inodes = [i for i in eachindex(global_node_numbers1) if global_node_numbers1[i] in inodes]
        ifes = FESetP1(local_inodes)
        VTK.vtkexportmesh("mbas002-interfaces-partition-$(partition).vtk", connasarray(ifes), fens1.xyz, VTK.P1;)
    end
    true
end
test()
nothing
end
