"""
Heat conduction example described by Amuthan A. Ramabathiran
http://www.codeproject.com/Articles/579983/Finite-Element-programming-in-Julia:
Unit cube, with known temperature distribution along the boundary,
and uniform heat generation rate inside.

Solution with domain decomposition. Sequential execution.
Version: 08/19/2024
"""
module Poisson2D_seq_examples
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
using FinEtoolsDDMethods.PartitionCoNCModule: rhs
using FinEtoolsDDMethods.DDCoNCSeqModule: partition_multiply!, preconditioner!, make_partitions
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
    
    t1 = time()
    cpartitioning, ncpartitions = cluster_partitioning(fens, fes, fes.label, nelperpart)
    @info("Number of clusters (coarse grid partitions): $(ncpartitions)")
        
    mor = CoNCData(fens, cpartitioning)
    Phi = transfmatrix(mor, LegendreBasis, nbf1max, Temp)
    Phi = Phi[fr, :]
    @info("Size of the reduced problem: $(size(Phi, 2))")
    @info("Transformation matrix: $(mebibytes(Phi)) [MiB]")
    @info "Generate clusters ($(round(time() - t1, digits=3)) [s])"
    
    t0 = time()
    function make_matrix(fes)
        femm2 = FEMMHeatDiff(IntegDomain(fes, volrule, 1.0), material)
        return conductivity(femm2, geom, Temp)
    end
    function make_interior_load(fes)
        femm2 = FEMMHeatDiff(IntegDomain(fes, volrule, 1.0), material)
        fi = ForceIntensity(Float64[Q])
        return distribloads(femm2, geom, Temp, fi, 3)
    end
    cpi = CoNCPartitioningInfo(fens, fes, nfpartitions, overlap, Temp) 
    partition_list = make_partitions(cpi, fes, make_matrix, make_interior_load)
    @info "Mean fine partition size = $(mean([partition_size(_p) for _p in partition_list]))"
    _b = mean([mebibytes(_p) for _p in partition_list])
    @info "Mean partition allocations: $(Int(round(_b, digits=0))) [MiB]" 
    @info "Total partition allocations: $(sum([mebibytes(_p) for _p in partition_list])) [MiB]" 
    @info "Create partitions time: $(time() - t1)"

    t1 = time()
    Kr_ff = spzeros(size(Phi, 2), size(Phi, 2))
    for partition in partition_list
        Kr_ff += (Phi' * partition.nonoverlapping_K * Phi)
    end
    Krfactor = lu(Kr_ff)
    Kr_ff = nothing
    GC.gc()
    @info "Create global factor: $(time() - t1)"
    @info("Global reduced factor: $(mebibytes(Krfactor)) [MiB]")

    t0 = time()
    F_f = rhs(partition_list)
    norm_F_f = norm(F_f)
    M! = preconditioner!(Krfactor, Phi, partition_list)
    (u_f, stats) = pcg_seq(
        (q, p) -> partition_multiply!(q, partition_list, p), 
        F_f, zeros(size(F_f));
        (M!)=(q, p) -> M!(q, p),
        # peeksolution=peeksolution,
        itmax=itmax, 
        # atol=0, rtol=relrestol, normtype = KSP_NORM_NATURAL
        atol= 0, rtol=relrestol, normtype = KSP_NORM_UNPRECONDITIONED
        )
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
end # module Poisson2D_examples
