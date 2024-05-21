module mbas001
using Test
using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsDomDecomp
using Metis
function test()
    ndoms = 3
    fens, fes = T3block(1.0, 1.0, 10, 13)
    femm1 = FEMMBase(IntegDomain(fes, SimplexRule(3, 1)))
    C = dualconnectionmatrix(femm1, fens, nodesperelem(boundaryfe(fes)))
    g = Metis.graph(C; check_hermitian=true)
    partitioning = Metis.partition(g, ndoms; alg=:KWAY)
    VTK.vtkexportmesh("mt014.vtk", connasarray(fes), fens.xyz, VTK.T3; scalars=[("partitioning", partitioning)])
    true
end
test()
nothing
end

