module fibers_soft_hard_hex_examples
using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtools.MeshTetrahedronModule: tetv
using FinEtoolsDeforLinear
using FinEtoolsDDParallel
using FinEtoolsDDParallel.CGModule: pcg_seq
using SymRCM
using Metis
using Test
using LinearAlgebra
using SparseArrays
using PlotlyLight
using Krylov
using LinearOperators
import Base: size, eltype
import LinearAlgebra: mul!
import CoNCMOR: CoNCData, transfmatrix, LegendreBasis
using Targe2
using DataDrop

function rotate(fens)
    Q = [cos(pi/2) -sin(pi/2); sin(pi/2) cos(pi/2)]
    for i in 1:count(fens)
        fens.xyz[i, :] .=  Q * fens.xyz[i, :]
    end
    fens 
end

function translate(fens, d)
    for i in 1:count(fens)
        fens.xyz[i, :] .+= d
    end
    fens
end

function matrix_unit(a, R, nL, nH, nW, tolerance)
    meshes = Array{Tuple{FENodeSet,AbstractFESet},1}()
    fens, fes = Q4elliphole(R, R, a + R, a + R, nL, nH, nW)
    push!(meshes, (fens, fes))
    fens, fes = Q4elliphole(R, R, a + R, a + R, nL, nH, nW)
    fens = rotate(fens)
    push!(meshes, (fens, fes))
    fens, fes = Q4elliphole(R, R, a + R, a + R, nL, nH, nW)
    fens = rotate(fens)
    fens = rotate(fens)
    push!(meshes, (fens, fes))
    fens, fes = Q4elliphole(R, R, a + R, a + R, nL, nH, nW)
    fens = rotate(fens)
    fens = rotate(fens)
    fens = rotate(fens)
    push!(meshes, (fens, fes))
    fens, fesa = mergenmeshes(meshes, tolerance)
    fes = fesa[1]
    for i in 2:length(fesa)
        fes = cat(fes, fesa[i])
    end
    setlabel!(fes, 1)
    return fens, fes
end

function fiber_unit(R, nr, tolerance)
    fens1, fes1 = Q4circlen(R, nr)
    fens1.xyz = fens1.xyz[:, 1:2]
    meshes = Array{Tuple{FENodeSet,AbstractFESet},1}()
    _fens, _fes = deepcopy(fens1), deepcopy(fes1)
    push!(meshes, (_fens, _fes))
    _fens, _fes = deepcopy(fens1), deepcopy(fes1)
    _fens = rotate(_fens)
    push!(meshes, (_fens, _fes))
    _fens, _fes = deepcopy(fens1), deepcopy(fes1)
    _fens = rotate(_fens)
    _fens = rotate(_fens)
    push!(meshes, (_fens, _fes))
    _fens, _fes = deepcopy(fens1), deepcopy(fes1)
    _fens = rotate(_fens)
    _fens = rotate(_fens)
    _fens = rotate(_fens)
    push!(meshes, (_fens, _fes))
    fens, fesa = mergenmeshes(meshes, tolerance)
    fes = fesa[1]
    for i in 2:length(fesa)
        fes = cat(fes, fesa[i])
    end
    setlabel!(fes, 2)
    return fens, fes
end

function fibers_mesh()
    refmultiplier = 2
    nr = 3
    a = 1.0
    R = 1.0 
    d = a + R + (nr - 1) * 2 * (a + R) + a + R
    nR = refmultiplier * 2 * 4
    nL = Int(nR / 2)
    nH = Int(nR / 2)
    nW = 2*Int(round(nr/2))
    Lz = 2.0 * d
    nlayers = refmultiplier * Int(ceil(Lz / 2))
    tolerance = R / nr / 100

    meshes = Array{Tuple{FENodeSet,AbstractFESet},1}()
    fens, fes = matrix_unit(a, R, nL, nH, nW, tolerance)
    push!(meshes, (fens, fes))
    fens, fes = fiber_unit(R, nR, tolerance)
    push!(meshes, (fens, fes))
    fens, fesa = mergenmeshes(meshes, tolerance)
    fes = fesa[1]
    for i in 2:length(fesa)
        fes = cat(fes, fesa[i])
    end
    fens = translate(fens, [(a + R), (a + R)])
    
    fens1, fes1 = deepcopy(fens), deepcopy(fes)
    
    meshes = Array{Tuple{FENodeSet,AbstractFESet},1}()
    for r in 1:nr
        for c in 1:nr
            fens, fes = deepcopy(fens1), deepcopy(fes1)
            fens = translate(fens, [(c - 1) * (a + R) * 2, (r - 1) * (a + R) * 2])
            push!(meshes, (fens, fes))
        end
    end

    fens, fesa = mergenmeshes(meshes, tolerance)
    fes = fesa[1]
    for i in 2:length(fesa)
        fes = cat(fes, fesa[i])
    end

    fens, fes = H8extrudeQ4(
        fens,
        fes,
        nlayers,
        (X, layer) -> [X[1], X[2], layer * Lz / nlayers],
    )

    File =  "fibers_soft_hard_hex-mesh.vtk"
    vtkexportmesh(File, fens, fes; scalars=[("label", fes.label)])
    # @async run(`"paraview.exe" $File`)

    return fens, fes
end # fibers_mesh

function test()
    # Isotropic material
    Em = 1.0
    num = 0.4999 
    num = 0.3 
    # Taylor data: nearly incompressible material
    Ef = 100000.0
    nuf = 0.3 # Compressible material
    CTE = 0.0
    magn = 1.0
    
    function getfrcL!(forceout, XYZ, tangents, feid, qpid)
        copyto!(forceout, [0.0; magn; 0.0; ])
    end
    
    nelperpart = 200
    nbf1max = 5

    fens, fes = fibers_mesh()
    println("Number of elements: $(count(fes))")

    matrixel = selectelem(fens, fes, label = 1)
    fiberel = selectelem(fens, fes, label = 2)

    File =  "fibers_mesh.vtk"
    vtkexportmesh(File, fens, fes; scalars=[("label", fes.label)])
    # @async run(`"paraview.exe" $File`)

    bfes = meshboundary(subset(fes, fiberel))
    sectionL = selectelem(fens, bfes; facing = true, direction = [0.0 0.0 +1.0])
    loadfes = subset(bfes, sectionL)
    # end cross-section surface  for the shear loading
    # sectionL = selectelem(fens, bfes; facing = true, direction = [0.0 0.0 +1.0])
    # 0 cross-section surface  for the reactions
    bfes = meshboundary(fes)
    section0 = selectelem(fens, bfes; facing = true, direction = [0.0 0.0 -1.0])

    MR = DeforModelRed3D
    matf = MatDeforElastIso(MR, 0.0, Ef, nuf, CTE)
    matm = MatDeforElastIso(MR, 0.0, Em, num, CTE)

    # Material orientation matrix
    csmat = [i == j ? one(Float64) : zero(Float64) for i = 1:3, j = 1:3]

    function updatecs!(csmatout, XYZ, tangents, feid, qpid)
        copyto!(csmatout, csmat)
    end

    geom = NodalField(fens.xyz)
    u = NodalField(zeros(size(fens.xyz, 1), 3)) # displacement field

    # Renumber the nodes
    femm = FEMMBase(IntegDomain(fes, GaussRule(3, 2)))
    C = connectionmatrix(femm, count(fens))
    perm = symrcm(C)
    
    lx0 = connectednodes(subset(bfes, section0))
    setebc!(u, lx0, true, 1, 0.0)
    setebc!(u, lx0, true, 2, 0.0)
    setebc!(u, lx0, true, 3, 0.0)
    applyebc!(u)
    numberdofs!(u, perm)
    fr = dofrange(u, DOF_KIND_FREE)
    dr = dofrange(u, DOF_KIND_DATA)
    
    println("nalldofs(u) = $(nalldofs(u))")
    println("nfreedofs(u) = $(nfreedofs(u))")

    fi = ForceIntensity(Float64, 3, getfrcL!)
    el2femm = FEMMBase(IntegDomain(loadfes, GaussRule(2, 2)))
    F = distribloads(el2femm, geom, u, fi, 2)
    F_f = F[fr]
    # femmm = FEMMDeforLinear(MR, IntegDomain(subset(fes, selectelem(fens, fes, label = 1)), TetRule(1)), matm)
    femmm = FEMMDeforLinearMSH8(MR, IntegDomain(subset(fes, matrixel), GaussRule(3, 2)), matm)
    associategeometry!(femmm, geom)
    K = stiffness(femmm, geom, u)
    # femmf = FEMMDeforLinear(MR, IntegDomain(subset(fes, selectelem(fens, fes, label = 2)), TetRule(1)), matf)
    femmf = FEMMDeforLinearMSH8(MR, IntegDomain(subset(fes, fiberel), GaussRule(3, 2)), matf)
    associategeometry!(femmf, geom)
    K += stiffness(femmf, geom, u)
    K_ff = K[fr, fr]

    U_f = K_ff \ F_f
    scattersysvec!(u, U_f)

    VTK.vtkexportmesh("fibers-sol.vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])   

    # partitioning = nodepartitioning(fens, npartitions)
    partitioning = zeros(Int, count(fens))
    femm = FEMMBase(IntegDomain(subset(fes, fiberel), GaussRule(3, 2)))
    C = dualconnectionmatrix(femm, fens, nodesperelem(boundaryfe(fes)))
    g = Metis.graph(C; check_hermitian=true)
    ndoms = Int(ceil(length(fiberel) / nelperpart))
    fiber_element_partitioning = Metis.partition(g, ndoms; alg=:KWAY)
    nfiberparts = maximum(fiber_element_partitioning)
    for e in eachindex(fiberel)
        for n in fes.conn[fiberel[e]]
            partitioning[n] = fiber_element_partitioning[e]
        end
    end
    femm = FEMMBase(IntegDomain(subset(fes, matrixel), GaussRule(3, 2)))
    C = dualconnectionmatrix(femm, fens, nodesperelem(boundaryfe(fes)))
    g = Metis.graph(C; check_hermitian=true)
    ndoms = Int(ceil(length(matrixel) / nelperpart))
    matrix_element_partitioning = Metis.partition(g, ndoms; alg=:KWAY)
    for e in eachindex(matrixel)
        for n in fes.conn[matrixel[e]]
            if partitioning[n] == 0
                partitioning[n] = matrix_element_partitioning[e] + nfiberparts
            end
        end
    end
    File =  "fibers-partitioning.vtk"
    partitionsfes = FESetP1(reshape(1:count(fens), count(fens), 1))
    vtkexportmesh(File, fens, partitionsfes; scalars=[("partition", partitioning)])
    # @async run(`"paraview.exe" $File`)
    
    @show npartitions = maximum(partitioning)

    mor = CoNCData(fens, partitioning)
    Phi = transfmatrix(mor, LegendreBasis, nbf1max, u)
    Phi = Phi[fr, :]
    transfm(m, t, tT) = (tT * m * t)
    transfv(v, t, tT) = (tT * v)
    PhiT = Phi'
    Kr_ff = transfm(K_ff, Phi, PhiT)
    @show size(Kr_ff)
    Krfactor = lu(Kr_ff)

    U_f = Phi * (Krfactor \ (PhiT * F_f))
    scattersysvec!(u, U_f)

    VTK.vtkexportmesh("fibers-red-sol.vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])   
    

    partitions = []
    for i in 1:npartitions
        pnl = findall(x -> x == i, partitioning)
        doflist = Int[]
        for n in pnl
            for d in axes(u.dofnums, 2)
                if u.dofnums[n, d] in fr
                    push!(doflist, u.dofnums[n, d])
                end
            end
        end
        pK = K[doflist, doflist]
        pKfactor = lu(pK)
        part = (nodelist = pnl, factor = pKfactor, doflist = doflist)
        push!(partitions, part)
    end

    function M!(q, p)
        q .= Phi * (Krfactor \ (PhiT * p))
        # rp = p - Phi * (PhiT * p)
        # q .= 0.0
        for part in partitions
            q[part.doflist] .+= (part.factor \ p[part.doflist])
        end
        q
    end

    (u_f, stats) = pcg_seq((q, p) -> mul!(q, K_ff, p), F_f, zeros(size(F_f));
        (M!)=(q, p) -> M!(q, p),
        itmax=1000, atol=1e-6 * norm(F_f), rtol=0)
    @show stats.niter
    DataDrop.store_json("fibers_soft_hard_hex-convergence" * ".json", stats)
    scattersysvec!(u, u_f)
    
    VTK.vtkexportmesh("fibers-hex-cg-sol.vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])


    true
end
test()
nothing
end
