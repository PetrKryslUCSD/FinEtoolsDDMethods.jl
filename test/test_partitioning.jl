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

