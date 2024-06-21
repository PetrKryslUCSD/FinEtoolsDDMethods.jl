module fibers_examples
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

function fibers_mesh_hex(ref = 1)
    nr = 3
    a = 1.0
    R = 1.0 
    d = a + R + (nr - 1) * 2 * (a + R) + a + R
    h = 2 * R / (2 + ref)
    nR = 2 * Int(round(R / h))
    nL = Int(round(nR / 2))
    nH = nL
    nW = 2 * Int(round(nR/2))
    Lz = 2.0 * d
    nlayers = ref + Int(ceil(Lz / (3 * h)))
    tolerance = R / nR / 100

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

function fibers_mesh_tet(ref = 1)
    nr = 3
    a = 1.0
    R = 1.0 
    h = 2 * R / (2 + ref)
    d = a + R + (nr - 1) * 2 * (a + R) + a + R
    Lz = 2.0 * d
    nlayers = ref + Int(ceil(Lz / (5 * h)))
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

function coarse_grid_partitioning(fens, fes, nelperpart, fiberel, matrixel)
    partitioning = zeros(Int, count(fens))
    femm = FEMMBase(IntegDomain(subset(fes, fiberel), PointRule()))
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
    femm = FEMMBase(IntegDomain(subset(fes, matrixel), PointRule()))
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
    npartitions = maximum(partitioning)
    partitioning, npartitions
end

function fine_grid_partitioning(fens, npartitions)
    partitioning = nodepartitioning(fens, npartitions)
    npartitions = maximum(partitioning)
    partitioning, npartitions
end

function _execute(label, kind, Em, num, Ef, nuf, nelperpart, nbf1max, nfpartitions, ref, mesh, boundary_rule, interior_rule, make_femm)
    CTE = 0.0
    magn = 1.0
    
    function getfrcL!(forceout, XYZ, tangents, feid, qpid)
        copyto!(forceout, [0.0; magn; 0.0])
    end

    println("Kind: $(string(kind))")
    println("Refinement multiplier: $(ref)")
    println("Number of elements per partition: $(nelperpart)")
    println("Number 1D basis functions: $(nbf1max)")
    println("Number fine grid partitions: $(nfpartitions)")
    
    fens, fes = mesh(ref)
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
    femm = FEMMBase(IntegDomain(fes, interior_rule))
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
    el2femm = FEMMBase(IntegDomain(loadfes, boundary_rule))
    F = distribloads(el2femm, geom, u, fi, 2)
    F_f = F[fr]
    femmm = make_femm(MR, IntegDomain(subset(fes, matrixel), interior_rule), matm)
    associategeometry!(femmm, geom)
    K = stiffness(femmm, geom, u)
    femmf = make_femm(MR, IntegDomain(subset(fes, fiberel), interior_rule), matf)
    associategeometry!(femmf, geom)
    K += stiffness(femmf, geom, u)
    K_ff = K[fr, fr]

    # U_f = K_ff \ F_f
    # scattersysvec!(u, U_f)

    # VTK.vtkexportmesh("fibers-tet-sol.vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])   

    cpartitioning, ncpartitions = coarse_grid_partitioning(fens, fes, nelperpart, fiberel, matrixel)
        
    # f = "fibers_soft_hard_$(string(kind))" *
    #     "-rm=$(ref)" *
    #     "-ne=$(nelperpart)" *
    #     "-n1=$(nbf1max)" * "-partitioning"
    # partitionsfes = FESetP1(reshape(1:count(fens), count(fens), 1))
    # vtkexportmesh(f * ".vtk", fens, partitionsfes; scalars=[("partition", cpartitioning)])
    # @async run(`"paraview.exe" $File`)
    
    mor = CoNCData(fens, cpartitioning)
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
    
    fpartitioning, nfpartitions = fine_grid_partitioning(fens, nfpartitions)
    
    partitions = []
    for i in 1:nfpartitions
        pnl = findall(x -> x == i, fpartitioning)
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
    norm_F_f = norm(F_f)
    (u_f, stats) = pcg_seq((q, p) -> mul!(q, K_ff, p), F_f, zeros(size(F_f));
        (M!)=(q, p) -> M!(q, p),
        itmax=1000, atol=1e-6 * norm_F_f, rtol=0)
    t1 = time()
    @show stats.niter
    stats = (niter = stats.niter, residuals = stats.residuals ./ norm_F_f)
    data = Dict(
        "nfreedofs_u" => nfreedofs(u),
        "ncpartitions" => ncpartitions,
        "nfpartitions" => nfpartitions,
        "size_Kr_ff" => size(Kr_ff),
        "stats" => stats,
        "time" => t1 - t0,
    )
    f = "fibers-$(label)-$(string(kind))" *
        "-rm=$(ref)" *
        "-ne=$(nelperpart)" *
        "-n1=$(nbf1max)"  * 
        "-nf=$(nfpartitions)" 
    DataDrop.store_json(f * ".json", data)
    scattersysvec!(u, u_f)
    
    f = "fibers-$(label)-$(string(kind))" *
        "-rm=$(ref)" *
        "-ne=$(nelperpart)" *
        "-n1=$(nbf1max)" * 
        "-nf=$(nfpartitions)"  * 
        "-cg-sol"
    VTK.vtkexportmesh(f * ".vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])
    
    true
end
# test(ref = 1)

function test(label = "soft_hard"; kind = "tet", Em = 1.0, num = 0.3, Ef = 1.20e5, nuf = 0.3, nelperpart = 200, nbf1max = 5, nfpartitions = 2, ref = 1)
    mesh = fibers_mesh_tet
    boundary_rule = TriRule(6)
    interior_rule = TetRule(4)
    make_femm = FEMMDeforLinearMST10
    if kind == "hex"
        mesh = fibers_mesh_hex
        boundary_rule = GaussRule(2, 2)
        interior_rule = GaussRule(3, 2)
        make_femm = FEMMDeforLinearMSH8
    end
    _execute(label, kind, Em, num, Ef, nuf, nelperpart, nbf1max, nfpartitions, ref, mesh, boundary_rule, interior_rule, make_femm)
end

nothing
end # module


