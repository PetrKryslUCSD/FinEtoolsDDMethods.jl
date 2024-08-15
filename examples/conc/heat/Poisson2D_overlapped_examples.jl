"""
Heat conduction example described by Amuthan A. Ramabathiran
http://www.codeproject.com/Articles/579983/Finite-Element-programming-in-Julia:
Unit cube, with known temperature distribution along the boundary,
and uniform heat generation rate inside.

Solution with domain decomposition.
Version: 08/03/2024
"""
module Poisson2D_overlapped_examples
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
    element_lists = PartitionCoNCDDMPIModule.subdomain_element_lists(fens, fes, nfpartitions, overlap)
    if visualize
        File = "Poisson2D_overlapped_examples-nonoverlapping-"
        for i in eachindex(element_lists)
            el = element_lists[i].nonoverlapping
            MeshExportModule.VTK.vtkexportmesh(File * "$(i)" * ".vtk", fens, subset(fes, el))
        end
        File = "Poisson2D_overlapped_examples-overlapping-"
        for i in eachindex(element_lists)
            el = element_lists[i].overlapping
            MeshExportModule.VTK.vtkexportmesh(File * "$(i)" * ".vtk", fens, subset(fes, el))
        end
    end
    # 
    node_lists = PartitionCoNCDDMPIModule.subdomain_node_lists(element_lists, fes)
    dof_lists = PartitionCoNCDDMPIModule.subdomain_dof_lists(node_lists, Temp.dofnums, fr)
    Kr_ff = spzeros(size(Phi_f, 2), size(Phi_f, 2))
    partition_matrices = []
    for i in eachindex(element_lists)
        el = element_lists[i].nonoverlapping
        femm1 = FEMMHeatDiff(IntegDomain(subset(fes, el), volrule, 1.0), material)
        K1 = conductivity(femm1, geom, Temp)
        el = setdiff(element_lists[i].overlapping, element_lists[i].nonoverlapping)
        femm2 = FEMMHeatDiff(IntegDomain(subset(fes, el), volrule, 1.0), material)
        K2 = conductivity(femm2, geom, Temp)
        Kn = K1
        Kn_ff = Kn[fr, fr]
        Kr_ff .+= Phi_f' * Kn_ff * Phi_f
        Ko = K1 + K2
        odof = dof_lists[i].overlapping
        Ko = Ko[odof, odof]
        @show odof
        push!(partition_matrices, (nonoverlapping = Kn_ff, overlapping = Ko, odof = odof))
    end
    Krfactor = lu(Kr_ff)

    # @show norm(Kr_ff - Phi_f' * K_ff * Phi_f) / norm(Kr_ff)

    # Ka = spzeros(size(K, 1), size(K, 2))
    # for i in eachindex(partition_matrices)
    #     Ka += partition_matrices[i].nonoverlapping
    # end
    # @show norm(K - Ka) / norm(K)

    t0 = time()
    norm_F_f = norm(F_f)
    function partition_multiply!(q, p)
        q .= zero(eltype(q))
        for i in eachindex(partition_matrices)
            q .+= partition_matrices[i].nonoverlapping * p
        end
        q
    end
    function precondition_solve!(q, p)
        q .= Phi_f * (Krfactor \ (Phi_f' * p))
        for i in eachindex(partition_matrices)
            d = partition_matrices[i].odof
            q[d] .+= (partition_matrices[i].overlapping \ p[d])
        end
        q
    end
    (u_f, stats) = pcg_seq((q, p) -> partition_multiply!(q, p), F_f, zeros(size(F_f));
        (M!)=(q, p) -> precondition_solve!(q, p),
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

function test(; kind = "Q8", N = 25, nbf1max = 2, nelperpart = 2*(nbf1max+1)^2, nfpartitions = 2, overlap = 1, itmax = 2000, relrestol = 1e-6, visualize = false)
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
