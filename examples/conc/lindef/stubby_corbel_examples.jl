module stubby_corbel
using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsDeforLinear
using FinEtoolsDDMethods
using FinEtoolsDDMethods.CGModule: pcg_seq
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

function test()
    # Isotropic material
    E = 1000.0
    nu = 0.4999 # Taylor data: nearly incompressible material
    nu = 0.3 # Compressible material
    W = 25.0
    H = 50.0
    L = 50.0
    htol = minimum([L, H, W]) / 1000
    uzex = -0.16
    magn = 0.2 * (-12.6) / 4
    Force = magn * W * H * 2
    CTE = 0.0
    n = 5 #

    function getfrcL!(forceout, XYZ, tangents, feid, qpid)
        copyto!(forceout, [0.0; 0.0; magn])
    end
    
    npartitions = 4
    nbf1max = 3

    fens, fes = H8block(W, L, H, n, 2 * n, 2 * n)
    println("Number of elements: $(count(fes))")
    bfes = meshboundary(fes)
    # end cross-section surface  for the shear loading
    sectionL = selectelem(fens, bfes; facing = true, direction = [0.0 +1.0 0.0])
    # 0 cross-section surface  for the reactions
    section0 = selectelem(fens, bfes; facing = true, direction = [0.0 -1.0 0.0])
    # 0 cross-section surface  for the reactions
    sectionlateral = selectelem(fens, bfes; facing = true, direction = [1.0 0.0 0.0])

    MR = DeforModelRed3D
    material = MatDeforElastIso(MR, 0.0, E, nu, CTE)

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
    
    femm = FEMMDeforLinearMSH8(MR, IntegDomain(fes, GaussRule(3, 2)), material)

    lx0 = connectednodes(subset(bfes, section0))
    setebc!(u, lx0, true, 1, 0.0)
    setebc!(u, lx0, true, 2, 0.0)
    setebc!(u, lx0, true, 3, 0.0)
    lx1 = connectednodes(subset(bfes, sectionlateral))
    setebc!(u, lx1, true, 1, 0.0)
    applyebc!(u)
    numberdofs!(u, perm)
    fr = dofrange(u, DOF_KIND_FREE)
    dr = dofrange(u, DOF_KIND_DATA)
    # numberdofs!(u)
    println("nfreedofs(u) = $(nfreedofs(u))")

    fi = ForceIntensity(Float64, 3, getfrcL!)
    el2femm = FEMMBase(IntegDomain(subset(bfes, sectionL), GaussRule(2, 2)))
    F = distribloads(el2femm, geom, u, fi, 2)
    F_f = F[fr]
    associategeometry!(femm, geom)
    K = stiffness(femm, geom, u)
    K_ff = K[fr, fr]

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
    
    VTK.vtkexportmesh("stubby_corbel-approx.vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])


    true
end
test()
nothing
end
