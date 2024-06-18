module fibrous_examples
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

function fibers_mesh()
    nr = 3
    a = 1.0
    R = 1.0 
    h = R / 2
    d = a + R + (nr - 1) * (a + 2 * R) + a + R
    Lz = 2.0 * d
    nlayers = Int(ceil(Lz / (9 * h)))
    input = """
    curve 1 line 0 0 $(d) 0
    curve 2 line $(d) 0 $(d) $(d)
    curve 3 line $(d) $(d) 0 $(d)
    curve 4 line 0 $(d) 0 0 
    """ 
    p = 1
    for r in 1:nr
        for c in 1:nr 
            cx = a + R + (r - 1) * (a + 2 * R)
            cy = a + R + (c - 1) * (a + 2 * R)
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

    X = zeros(4, 3)
    for i in eachindex(fes)
        X[1, :] = fens.xyz[fes.conn[i][1], :]
        X[2, :] = fens.xyz[fes.conn[i][2], :]
        X[3, :] = fens.xyz[fes.conn[i][3], :]
        X[4, :] = fens.xyz[fes.conn[i][4], :]
        @test  tetv(X) > 0.0
    end

    fens, fes = T4toT10(fens, fes)

    # File =  "fibers_mesh.vtk"
    # vtkexportmesh(File, fens, fes; scalars=[("label", fes.label)])
    # @async run(`"paraview.exe" $File`)

    return fens, fes
end # fibers_mesh

function test()
    # Isotropic material
    Em = 1000.0
    num = 0.4999 # Taylor data: nearly incompressible material
    Ef = 100000.0
    nuf = 0.3 # Compressible material
    CTE = 0.0
    magn = 1.0
    
    function getfrcL!(forceout, XYZ, tangents, feid, qpid)
        copyto!(forceout, [0.0; magn; 0.0])
    end
    
    npartitions = 8
    nbf1max = 2

    fens, fes = fibers_mesh()
    println("Number of elements: $(count(fes))")

    File =  "fibers_mesh.vtk"
    vtkexportmesh(File, fens, fes; scalars=[("label", fes.label)])
    # @async run(`"paraview.exe" $File`)

    bfes = meshboundary(fes)
    # end cross-section surface  for the shear loading
    sectionL = selectelem(fens, bfes; facing = true, direction = [0.0 0.0 +1.0])
    # 0 cross-section surface  for the reactions
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
    
    println("nalldofs(u) = $(nalldofs(u))")
    println("nfreedofs(u) = $(nfreedofs(u))")

    fi = ForceIntensity(Float64, 3, getfrcL!)
    el2femm = FEMMBase(IntegDomain(subset(bfes, sectionL), GaussRule(2, 2)))
    F = distribloads(el2femm, geom, u, fi, 2)
    F_f = F[fr]
    # femmm = FEMMDeforLinear(MR, IntegDomain(subset(fes, selectelem(fens, fes, label = 1)), TetRule(1)), matm)
    femmm = FEMMDeforLinearMST10(MR, IntegDomain(subset(fes, selectelem(fens, fes, label = 1)), TetRule(4)), matm)
    associategeometry!(femmm, geom)
    K = stiffness(femmm, geom, u)
    # femmf = FEMMDeforLinear(MR, IntegDomain(subset(fes, selectelem(fens, fes, label = 2)), TetRule(1)), matf)
    femmf = FEMMDeforLinearMST10(MR, IntegDomain(subset(fes, selectelem(fens, fes, label = 2)), TetRule(4)), matf)
    associategeometry!(femmf, geom)
    K += stiffness(femmf, geom, u)
    K_ff = K[fr, fr]

    U_f = K_ff \ F_f
    scattersysvec!(u, U_f)

    VTK.vtkexportmesh("fibers-sol.vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])   

    partitioning = nodepartitioning(fens, npartitions)
    partitionnumbers = unique(partitioning)
    npartitions = length(partitionnumbers)

    mor = CoNCData(fens, partitioning)
    Phi = transfmatrix(mor, LegendreBasis, nbf1max, u)
    Phi = Phi[fr, :]
    transfm(m, t, tT) = (tT * m * t)
    transfv(v, t, tT) = (tT * v)
    PhiT = Phi'
    Kr_ff = transfm(K_ff, Phi, PhiT)
    @show size(Kr_ff)
    Krfactor = lu(Kr_ff)
    

    partitions = []
    for i in partitionnumbers
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

    @show length(partitions)

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
        itmax=1000, atol=1e-10, rtol=1e-10)
    @show stats
    scattersysvec!(u, u_f)
    
    VTK.vtkexportmesh("fibers-cg-sol.vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])


    true
end
test()
nothing
end
