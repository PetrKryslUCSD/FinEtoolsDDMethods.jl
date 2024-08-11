module simple_block_condition_examples
using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtools.MeshExportModule: CSV
using FinEtools.MeshTetrahedronModule: tetv
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
using Targe2
using DataDrop
using Statistics
using Arpack

function _execute(kind, E, nu, nelperpart, nbf1max, nfpartitions, ref, itmax, relrestol, visualize)
    H = 4.0
    W = 5.0
    L = 30.0
    nH, nW, nL = 4, 5, 30
    CTE = 0.0
    magn = 1.0
    rho = 1.0
    E = Float64(E)
    nu = Float64(nu)
    
    function getfrcL!(forceout, XYZ, tangents, feid, qpid)
        copyto!(forceout, [0.0; magn; 0.0])
    end

    if kind == "hex"
        mesh = H8block
        boundary_rule = GaussRule(2, 2)
        interior_rule = GaussRule(3, 2)
        make_femm = FEMMDeforLinearMSH8
    elseif kind == "tet"
        mesh = T10block
        boundary_rule = TriRule(6)
        interior_rule = TetRule(4)
        make_femm = FEMMDeforLinearMST10
    end
    
    fens, fes = mesh(H, W, L, ref*nH, ref*nW, ref*nL)

    MR = DeforModelRed3D
    mat1 = MatDeforElastIso(MR, rho, E, nu, CTE)

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
    
    bfes = meshboundary(fes)
    sectionL = selectelem(fens, bfes; facing = true, direction = [0.0 0.0 +1.0])
    loadfes = subset(bfes, sectionL)
    # end cross-section surface  for the shear loading
    # sectionL = selectelem(fens, bfes; facing = true, direction = [0.0 0.0 +1.0])
    # 0 cross-section surface  for the reactions
    bfes = meshboundary(fes)
    section0 = selectelem(fens, bfes; facing = true, direction = [0.0 0.0 -1.0])

    lx0 = connectednodes(subset(bfes, section0))
    setebc!(u, lx0, true, 1, 0.0)
    setebc!(u, lx0, true, 2, 0.0)
    setebc!(u, lx0, true, 3, 0.0)
    applyebc!(u)
    numberdofs!(u, perm)
    fr = dofrange(u, DOF_KIND_FREE)
    dr = dofrange(u, DOF_KIND_DATA)
    
    println("Kind: $(string(kind))")
    println("Material: $(E), $(nu)")
    println("Refinement factor: $(ref)")
    println("Number of elements per partition: $(nelperpart)")
    println("Number of 1D basis functions: $(nbf1max)")
    println("Number of elements: $(count(fes))")
    println("Number of free dofs = $(nfreedofs(u))")

    fi = ForceIntensity(Float64, 3, getfrcL!)
    el2femm = FEMMBase(IntegDomain(loadfes, boundary_rule))
    F = distribloads(el2femm, geom, u, fi, 2)
    F_f = F[fr]
    femm = make_femm(MR, IntegDomain(fes, interior_rule), mat1)
    associategeometry!(femm, geom)
    K = stiffness(femm, geom, u)
    K_ff = K[fr, fr]
    M = mass(femm, geom, u)
    M_ff = M[fr, fr]
    M_ff = diagm(ones(size(M_ff, 1))) 

    cpartitioning, ncpartitions = FinEtoolsDDMethods.cluster_partitioning(fens, fes, fes.label, nelperpart)
    println("Number of clusters (coarse grid partitions): $(ncpartitions)")
 
    mor = CoNCData(fens, cpartitioning)
    Phi = transfmatrix(mor, LegendreBasis, nbf1max, u)
    Phi = Phi[fr, :]
    transfm(m, t, tT) = (tT * m * t)
    transfv(v, t, tT) = (tT * v)
    PhiT = Phi'
    Kr_ff = transfm(K_ff, Phi, PhiT)
    Mr_ff = transfm(M_ff, Phi, PhiT)
    println("Size of the reduced problem: $(size(Kr_ff))")
    
    neigvs = 20

    d, v, nconv = Arpack.eigs(
        # Symmetric(K_ff);
        Symmetric(K_ff), Symmetric(M_ff);
        nev=neigvs,
        which=:SM,
        explicittransform=:none,
        check=1,
    )
    @show d
    vectors = []
    for i = 1:neigvs
        scattersysvec!(u, v[:, i])
        push!(vectors, ("Mode_$(i)_$(d[i])", deepcopy(u.values)))
    end
    File = "simple_block_condition-full" *
    "-rf=$(ref)" *
    ".vtk"
    vtkexportmesh(
        File, fens, fes;
        vectors = vectors,
    )

    d, v, nconv = Arpack.eigs(
        # Symmetric(Kr_ff);
        Symmetric(Kr_ff), Symmetric(Mr_ff);
        nev=neigvs,
        which=:SM,
        explicittransform=:none,
        check=1,
    )
    @show d
    vectors = []
    for i = 1:neigvs
        scattersysvec!(u, Phi * v[:, i])
        push!(vectors, ("Mode_$(i)_$(d[i])", deepcopy(u.values)))
    end
    File = "simple_block_condition-red" *
    "-rf=$(ref)" *
    "-ne=$(nelperpart)" *
    "-n1=$(nbf1max)" *
    ".vtk"
    vtkexportmesh(
        File, fens, fes;
        vectors = vectors,
    )

    if visualize
        f = "fibers-$(label)-$(string(kind))" *
            "-rf=$(ref)" *
            "-ne=$(nelperpart)" *
            "-n1=$(nbf1max)" *
            "-nf=$(nfpartitions)" *
            "-cg-sol"
        VTK.vtkexportmesh(f * ".vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])
        
        p = 1
        for nodelist in nodelists
            cel = connectedelems(fes, nodelist, count(fens))
            f = "fibers-$(label)-$(string(kind))" *
            "-rf=$(ref)" *
            "-nf=$(nfpartitions)" *
            "-p=$p"
            vtkexportmesh(f * ".vtk", fens, subset(fes, cel))
            p += 1
        end
    end
    
    true
end
# test(ref = 1)

function test(; kind = "hex", E = 1.0e3, nu = 0.4999, nelperpart = 200, nbf1max = 5, nfpartitions = 2, overlap = 1, ref = 1, itmax = 2000, relrestol = 1e-6, visualize = false)
    _execute(kind, E, nu, nelperpart, nbf1max, nfpartitions, ref, itmax, relrestol, visualize)
end

nothing
end # module


