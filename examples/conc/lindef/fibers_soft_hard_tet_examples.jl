module fibres_soft_hard_tet_examples
using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtools.MeshExportModule: CSV
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

function fibers_mesh(refmultiplier = 1)
    nr = 3
    a = 1.0
    R = 1.0 
    h = R / 2 / refmultiplier
    d = a + R + (nr - 1) * 2 * (a + R) + a + R
    Lz = 2.0 * d
    nlayers = refmultiplier * Int(ceil(Lz / (5 * h)))
    input = """
    curve 1 line 0 0 $(d) 0
    curve 2 line $(d) 0 $(d) $(d)
    curve 3 line $(d) $(d) 0 $(d)  
    curve 4 line 0 $(d) 0 0 
    """ 
    p = 1
    for r in 1:nr
        for c in 1:nr 
            cx = a + R + (r - 1) * 2 * (a + R)
            cy = a + R + (c - 1) * 2 * (a + R)
            input *= "\n" * "curve $(4+p) circle center $(cx) $(cy) radius $R"
            p += 1
        end
    end
    input *= "\n" * 
    """
    subregion 1  property 1 boundary 1 2 3 4 hole """
    p = 1
    for r in 1:nr
        for c in 1:nr
            input *= " $(-(4+p))"
            p += 1
        end
    end
    input *= "\n" * 
    """
    subregion 2  property 2 boundary """
    p = 1
    for r in 1:nr
        for c in 1:nr
            input *= " $(+(4+p))"
            p += 1
        end
    end
    input *= "\n" * 
    """    
    m-ctl-point constant $h
    """
    
    mesh = triangulate(input)
    
    fens = FENodeSet(mesh.xy)
    fes = FESetT3(mesh.triconn)
    label = fill(0, count(fes))
    for i in eachindex(mesh.trigroups)
        for j in mesh.trigroups[i]
            label[j] = i
        end
    end
    setlabel!(fes, label)

    fens, fes = T4extrudeT3(
        fens,
        fes,
        nlayers,
        (X, layer) -> [X[1], X[2], layer * Lz / nlayers],
    )

    # X = zeros(4, 3)
    # for i in eachindex(fes)
    #     X[1, :] = fens.xyz[fes.conn[i][1], :]
    #     X[2, :] = fens.xyz[fes.conn[i][2], :]
    #     X[3, :] = fens.xyz[fes.conn[i][3], :]
    #     X[4, :] = fens.xyz[fes.conn[i][4], :]
    #     @test  tetv(X) > 0.0
    # end

    fens, fes = T4toT10(fens, fes)

    # File =  "fibers_mesh.vtk"
    # vtkexportmesh(File, fens, fes; scalars=[("label", fes.label)])
    # @async run(`"paraview.exe" $File`)

    return fens, fes
end # fibers_mesh

function test(; nelperpart = 200, nbf1max = 5, refmultiplier = 2)
    # Isotropic material
    Em = 1.0
    # num = 0.4999 
    num = 0.3 
    # Taylor data: nearly incompressible material
    Ef = 100000.0
    nuf = 0.3 # Compressible material
    CTE = 0.0
    magn = 1.0
    
    function getfrcL!(forceout, XYZ, tangents, feid, qpid)
        copyto!(forceout, [0.0; magn; 0.0])
    end

    println("Number of elements per partition: $(nelperpart)")
    println("Number 1D basis functions: $(nbf1max)")
    println("Refinement multiplier: $(refmultiplier)")
    
    fens, fes = fibers_mesh(refmultiplier)
    println("Number of elements: $(count(fes))")

    matrixel = selectelem(fens, fes, label = 1)
    fiberel = selectelem(fens, fes, label = 2)

    # File =  "fibers_mesh.vtk"
    # vtkexportmesh(File, fens, fes; scalars=[("label", fes.label)])
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
    femm = FEMMBase(IntegDomain(fes, TetRule(1)))
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
    
    println("nfreedofs(u) = $(nfreedofs(u))")

    fi = ForceIntensity(Float64, 3, getfrcL!)
    el2femm = FEMMBase(IntegDomain(loadfes, TriRule(6)))
    F = distribloads(el2femm, geom, u, fi, 2)
    F_f = F[fr]
    femmm = FEMMDeforLinearMST10(MR, IntegDomain(subset(fes, matrixel), TetRule(4)), matm)
    associategeometry!(femmm, geom)
    K = stiffness(femmm, geom, u)
    femmf = FEMMDeforLinearMST10(MR, IntegDomain(subset(fes, fiberel), TetRule(4)), matf)
    associategeometry!(femmf, geom)
    K += stiffness(femmf, geom, u)
    K_ff = K[fr, fr]

    # U_f = K_ff \ F_f
    # scattersysvec!(u, U_f)

    # VTK.vtkexportmesh("fibers-tet-sol.vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])   

    # partitioning = nodepartitioning(fens, npartitions)
    partitioning = zeros(Int, count(fens))
    femm = FEMMBase(IntegDomain(subset(fes, fiberel), TetRule(4)))
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
    femm = FEMMBase(IntegDomain(subset(fes, matrixel), TetRule(4)))
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

    # U_f = Phi * (Krfactor \ (PhiT * F_f))
    # scattersysvec!(u, U_f)

    # VTK.vtkexportmesh("fibers-tet-red-sol.vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])   
    

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
        for part in partitions
            q[part.doflist] .+= (part.factor \ p[part.doflist])
        end
        q
    end

    t0 = time()
    (u_f, stats) = pcg_seq((q, p) -> mul!(q, K_ff, p), F_f, zeros(size(F_f));
        (M!)=(q, p) -> M!(q, p),
        itmax=1000, atol=1e-6 * norm(F_f), rtol=0)
    t1 = time()
    @show stats.niter
    data = Dict(
        "nfreedofs_u" => nfreedofs(u),
        "npartitions" => npartitions,
        "size_Kr_ff" => size(Kr_ff),
        "stats" => stats,
        "time" => t1 - t0,
    )
    f = "fibers_soft_hard_tet" *
        "-rm=$(refmultiplier)" *
        "-ne=$(nelperpart)" *
        "-n1=$(nbf1max)" 
    DataDrop.store_json(f * ".json", data)
    scattersysvec!(u, u_f)
    
    VTK.vtkexportmesh("fibers-tet-cg-sol.vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])


    true
end
# test(refmultiplier = 1)
nothing
end
