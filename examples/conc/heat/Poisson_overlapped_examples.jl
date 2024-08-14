"""
Heat conduction example described by Amuthan A. Ramabathiran
http://www.codeproject.com/Articles/579983/Finite-Element-programming-in-Julia:
Unit cube, with known temperature distribution along the boundary,
and uniform heat generation rate inside.

Solution with domain decomposition.
Version: 08/03/2024
"""
module Poisson_overlapped_examples
using FinEtools
using FinEtools.AlgoBaseModule: solve_blocked!, matrix_blocked, vector_blocked
using FinEtools.AssemblyModule
using FinEtools.MeshExportModule
using FinEtoolsHeatDiff
using LinearAlgebra
using DataDrop
using SparseArrays
using SymRCM
import CoNCMOR: CoNCData, transfmatrix, LegendreBasis
using FinEtoolsDDMethods
using FinEtoolsDDMethods.CGModule: pcg_seq
using Statistics

function _execute(N, mesher, volrule, nelperpart, nbf1max, nfpartitions, overlap, itmax, relrestol, visualize)
    # @info("""
    # Heat conduction example described by Amuthan A. Ramabathiran
    # http://www.codeproject.com/Articles/579983/Finite-Element-programming-in-Julia:
    # Unit cube, with known temperature distribution along the boundary,
    # and uniform heat generation rate inside.  Mesh of regular quadratic HEXAHEDRA,
    # in a grid of $(N) x $(N) x $(N) edges.
    # Version: 08/13/2024
    # """)
    A = 1.0 # dimension of the domain (length of the side of the square)
    thermal_conductivity = [i == j ? one(Float64) : zero(Float64) for i = 1:3, j = 1:3] # conductivity matrix
    Q = -6.0 # internal heat generation rate
    function getsource!(forceout, XYZ, tangents, feid, qpid)
        forceout[1] = Q #heat source
    end
    tempf(x) = (1.0 .+ x[:, 1] .^ 2 + 2.0 .* x[:, 2] .^ 2)#the exact distribution of temperature1
    t0 = time()
    fens, fes = mesher(A, A, A, N, N, N)
    @info("Mesh generation ($(round(time() - t0, digits=3)) [s])")
    t0 = time()
    geom = NodalField(fens.xyz)
    Temp = NodalField(zeros(size(fens.xyz, 1), 1))
    Tolerance = 1.0 / N / 100.0
    l1 = selectnode(fens; box = [0.0 0.0 0.0 A 0.0 A], inflate = Tolerance)
    l2 = selectnode(fens; box = [A A 0.0 A 0.0 A], inflate = Tolerance)
    l3 = selectnode(fens; box = [0.0 A 0.0 0.0 0.0 A], inflate = Tolerance)
    l4 = selectnode(fens; box = [0.0 A A A 0.0 A], inflate = Tolerance)
    l5 = selectnode(fens; box = [0.0 A 0.0 A 0.0 0.0], inflate = Tolerance)
    l6 = selectnode(fens; box = [0.0 A 0.0 A A A], inflate = Tolerance)
    List = vcat(l1, l2, l3, l4, l5, l6)
    setebc!(Temp, List, true, 1, tempf(geom.values[List, :])[:])
    numberdofs!(Temp)
    fr = dofrange(Temp, DOF_KIND_FREE)
    dr = dofrange(Temp, DOF_KIND_DATA)
    @info("Number of free degrees of freedom: $(nfreedofs(Temp)) ($(round(time() - t0, digits=3)) [s])")
    t0 = time()
    material = MatHeatDiff(thermal_conductivity)
    femm = FEMMHeatDiff(IntegDomain(fes, volrule), material)
    K = conductivity(femm, geom, Temp)
    @info("Conductivity ($(round(time() - t0, digits=3)) [s])")
    K_ff = K[fr, fr]
    t0 = time()
    fi = ForceIntensity(Float64[Q])
    F1 = distribloads(femm, geom, Temp, fi, 3)
    @info("Internal heat generation ($(round(time() - t0, digits=3)) [s])")
    t0 = time()
    T_d = gathersysvec(Temp, DOF_KIND_DATA)
    F_f = (F1 .- K[:, dr] * T_d)[fr]
    @info("Right hand side ($(round(time() - t0, digits=3)) [s])")
    
    t0 = time()
    cpartitioning, ncpartitions = FinEtoolsDDMethods.cluster_partitioning(fens, fes, fes.label, nelperpart)
    @info("Number of clusters (coarse grid partitions): $(ncpartitions)")
    mor = CoNCData(fens, cpartitioning)
    Phi = transfmatrix(mor, LegendreBasis, nbf1max, Temp)
    Phi_f = Phi[fr, :]
    @info("Size of the reduced problem: $(size(Phi_f, 2))")
    @info "Clustering ($(round(time() - t0, digits=3)) [s])"

    t0 = time()
    fpartitions = fine_grid_partitions(fens, fes, nfpartitions, overlap, Temp.dofnums, fr)
    meansize = mean([length(part.doflist) for part in fpartitions])
    @info("Mean fine partition size: $(meansize)")
    @info "Fine grid partitioning ($(round(time() - t0, digits=3)) [s])"
    t0 = time()
    M! = preconditioner(fpartitions, Phi_f, K_ff)
    @info "Preconditioner setup ($(round(time() - t0, digits=3)) [s])"

    t0 = time()
    norm_F_f = norm(F_f)
    (u_f, stats) = pcg_seq((q, p) -> mul!(q, K_ff, p), F_f, zeros(size(F_f));
        (M!)=(q, p) -> M!(q, p),
        itmax=itmax, atol=relrestol * norm_F_f, rtol=0)
    @info("Number of iterations:  $(stats.niter)")
    stats = (niter = stats.niter, residuals = stats.residuals ./ norm_F_f)
    scattersysvec!(Temp, u_f, DOF_KIND_FREE)
    @info("Solution ($(round(time() - t0, digits=3)) [s])")
    t0 = time()
    Error = 0.0
    for k in axes(fens.xyz, 1)
        Error = Error + abs.(Temp.values[k, 1] - tempf(reshape(fens.xyz[k, :], (1, 3)))[1])
    end
    @info("Error =$Error ($(round(time() - t0, digits=3)) [s])")
    # File =  "a.vtk"
    # MeshExportModule.vtkexportmesh (File, fes.conn, [geom.values Temp.values], MeshExportModule.T3; scalars=Temp.values, scalars_name ="Temperature")
    true
end # _execute

function test(; kind = "H20", N = 25, nbf1max = 2, nelperpart = 8*(nbf1max+1)^3, nfpartitions = 2, overlap = 1, itmax = 2000, relrestol = 1e-6, visualize = false)
    if kind == "H20"
        mesher = H20block
        volrule = GaussRule(3, 3)
    elseif kind == "T10"
        mesher = T10block
        volrule = TetRule(4)
    elseif kind == "T4"
        mesher = T4block
        volrule = TetRule(1)
    else
        error("Unknown kind of element")
    end
    _execute(N, mesher, volrule, nelperpart, nbf1max, nfpartitions, overlap, itmax, relrestol, visualize)
end

nothing
end # module Poisson_overlapped_examples
