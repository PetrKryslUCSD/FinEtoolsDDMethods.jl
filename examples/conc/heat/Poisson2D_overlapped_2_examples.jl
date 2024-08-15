"""
Heat conduction example described by Amuthan A. Ramabathiran
http://www.codeproject.com/Articles/579983/Finite-Element-programming-in-Julia:
Unit cube, with known temperature distribution along the boundary,
and uniform heat generation rate inside.

Solution with domain decomposition.
Version: 08/03/2024
"""
module Poisson2D_overlapped_2_examples
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
using FinEtoolsDDMethods.PartitionCoNCDDMPIModule
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
    thermal_conductivity = [i == j ? one(Float64) : zero(Float64) for i = 1:2, j = 1:2] # conductivity matrix
    Q = -6.0 # internal heat generation rate
    function getsource!(forceout, XYZ, tangents, feid, qpid)
        forceout[1] = Q #heat source
    end
    tempf(x) = (1.0 .+ x[:, 1] .^ 2 + 2.0 .* x[:, 2] .^ 2)#the exact distribution of temperature1
    t0 = time()
    fens, fes = mesher(A, A, N, N)
    @info("Mesh generation ($(round(time() - t0, digits=3)) [s])")
    t0 = time()
    geom = NodalField(fens.xyz)
    Temp = NodalField(zeros(size(fens.xyz, 1), 1))
    Tolerance = 1.0 / N / 100.0
    l1 = selectnode(fens; box = [0.0 0.0 0.0 A], inflate = Tolerance)
    l2 = selectnode(fens; box = [A   A   0.0 A], inflate = Tolerance)
    l3 = selectnode(fens; box = [0.0 A 0.0 0.0], inflate = Tolerance)
    l4 = selectnode(fens; box = [0.0 A  A  A], inflate = Tolerance)
    List = vcat(l1, l2, l3, l4, )
    setebc!(Temp, List, true, 1, tempf(geom.values[List, :])[:])
    numberdofs!(Temp)
    fr = dofrange(Temp, DOF_KIND_FREE)
    dr = dofrange(Temp, DOF_KIND_DATA)
    @info("Number of free degrees of freedom: $(nfreedofs(Temp)) ($(round(time() - t0, digits=3)) [s])")
    t0 = time()
    material = MatHeatDiff(thermal_conductivity)
    femm = FEMMHeatDiff(IntegDomain(fes, volrule, 1.0), material)
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
        Error = Error + abs.(Temp.values[k, 1] - tempf(reshape(fens.xyz[k, :], (1, 2)))[1])
    end
    @info("Error =$Error ($(round(time() - t0, digits=3)) [s])")

    
    t0 = time()
    function make_matrix(fes)
        femm2 = FEMMHeatDiff(IntegDomain(fes, volrule, 1.0), material)
        return conductivity(femm2, geom, Temp)
    end
    cp = PartitionCoNCDDMPIModule.CoNCPartitioning(fens, fes, nfpartitions, overlap, make_matrix, Temp, Phi) 
    @info("Create partitioning ($(round(time() - t0, digits=3)) [s])")

    t0 = time()
    norm_F_f = norm(F_f)
    (u_f, stats) = pcg_seq((q, p) -> PartitionCoNCDDMPIModule.partition_multiply!(q, cp, p), F_f, zeros(size(F_f));
        (M!)=(q, p) -> PartitionCoNCDDMPIModule.precondition_solve!(q, cp, p),
        itmax=itmax, atol=relrestol * norm_F_f, rtol=0)
    @info("Number of iterations:  $(stats.niter)")
    stats = (niter = stats.niter, residuals = stats.residuals ./ norm_F_f)
    scattersysvec!(Temp, u_f, DOF_KIND_FREE)
    @info("Solution ($(round(time() - t0, digits=3)) [s])")

    # if visualize
    #     geometry = xyz3(fens)
    #     geometry[:, 3] .= Temp.values
    #     fens.xyz = geometry
    #     File = "Poisson2D_overlapped_examples-sol.vtk"
    #     MeshExportModule.VTK.vtkexportmesh(File, fens, fes; scalars=[("Temp", Temp.values)])
    # end
    true
end # _execute

function test(; kind = "Q8", N = 25, nbf1max = 2, nelperpart = 2*(nbf1max+1)^2, nfpartitions = 2, overlap = 1, itmax = 1000, relrestol = 1e-6, visualize = false)
    if kind == "Q8"
        mesher = Q8block
        volrule = GaussRule(2, 3)
    elseif kind == "T6"
        mesher = T6block
        volrule = TriRule(3)
    elseif kind == "T3"
        mesher = T3block
        volrule = TriRule(1)
    else
        error("Unknown kind of element")
    end
    _execute(N, mesher, volrule, nelperpart, nbf1max, nfpartitions, overlap, itmax, relrestol, visualize)
end

nothing
end # module Poisson2D_overlapped_examples
