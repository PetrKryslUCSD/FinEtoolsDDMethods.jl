"""
Heat conduction example described by Amuthan A. Ramabathiran
http://www.codeproject.com/Articles/579983/Finite-Element-programming-in-Julia:
Unit cube, with known temperature distribution along the boundary,
and uniform heat generation rate inside.

Solution with domain decomposition and MPI.
Version: 08/19/2024
"""
module Poisson2D_mpi_driver
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
using FinEtoolsDDMethods.CGModule: pcg_mpi_2level_Schwarz
using FinEtoolsDDMethods.PartitionCoNCDDMPIModule
using Statistics
using MPI

function peeksolution(iter, x, resnorm)
    @info("Iteration: $(iter), Residual norm: $(resnorm)")
end

function _execute(N, mesher, volrule, nelperpart, nbf1max, overlap, itmax, relrestol, visualize)
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
    tempf(x) = (1.0 .+ x[:, 1] .^ 2 + 2.0 .* x[:, 2] .^ 2)#the exact distribution of temperature
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    nfpartitions = nprocs - 1

    rank == 0 && (@info "Number of processes: $nprocs")
    rank == 0 && (@info "Number of partitions: $nfpartitions")

    t1 = time()
    fens, fes = mesher(A, A, N, N)
    rank == 0 && @info("Mesh generation ($(round(time() - t1, digits=3)) [s])")
    t1 = time()
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

    material = MatHeatDiff(thermal_conductivity)

    t1 = time()
    cpartitioning, ncpartitions = FinEtoolsDDMethods.cluster_partitioning(fens, fes, fes.label, nelperpart)
    rank == 0 && @info("Number of clusters (coarse grid partitions): $(ncpartitions)")
    mor = CoNCData(fens, cpartitioning)
    Phi = transfmatrix(mor, LegendreBasis, nbf1max, Temp)
    Phi = Phi[freedofs(Temp), :]
    rank == 0 && @info("Size of the reduced problem: $(size(Phi, 2))")
    rank == 0 && @info "Clustering ($(round(time() - t1, digits=3)) [s])"

    function make_matrix(fes)
        femm2 = FEMMHeatDiff(IntegDomain(fes, volrule, 1.0), material)
        return conductivity(femm2, geom, Temp)
    end

    function make_interior_load(fes)
        femm2 = FEMMHeatDiff(IntegDomain(fes, volrule, 1.0), material)
        fi = ForceIntensity(Float64[Q])
        return distribloads(femm2, geom, Temp, fi, 3)
    end

    t1 = time()
    partition = nothing
    if rank > 0
        cpi = CoNCPartitioningInfo(fens, fes, nfpartitions, overlap, Temp) 
        partition = CoNCPartitionData(cpi, rank, fes, Phi, make_matrix, make_interior_load)
    end    
    MPI.Barrier(comm)
    rank == 0 && (@info "Create partitions time: $(time() - t1)")

    t1 = time()
    F_f = zeros(nfreedofs(Temp))
    if rank > 0
        F_f .= partition.rhs
    end
    MPI.Reduce!(F_f, MPI.SUM, comm; root=0) # Reduce the data-determined rhs
    rank == 0 && (@info "Compute RHS: $(time() - t1)")
    
    t1 = time()
    Kr_ff = spzeros(size(Phi, 2), size(Phi, 2))
    if rank > 0
        Kr_ff = partition.reduced_K
    end
    ks = MPI.gather(Kr_ff, comm; root=0)
    if rank == 0
        for k in ks
            Kr_ff += k
        end
        Krfactor = lu(Kr_ff)
    end
    rank == 0 && (@info "Create global factor: $(time() - t1)")
    
    t1 = time()
    norm_F_f = norm(F_f) 
    (u_f, stats) = pcg_mpi_2level_Schwarz(
        comm, 
        rank,
        (q, p) -> partition_multiply!(q, partition, p),
        F_f,
        zeros(size(F_f)),
        (q, p) -> precondition_global_solve!(q, Krfactor, Phi, p), 
        (q, p) -> precondition_local_solve!(q, partition, p);
        itmax=itmax, atol=relrestol * norm_F_f, rtol=0,
        peeksolution=peeksolution)

    rank == 0 && @info("Number of iterations:  $(stats.niter)")
    stats = (niter=stats.niter, resnorm=stats.resnorm ./ norm_F_f)
    scattersysvec!(Temp, u_f, DOF_KIND_FREE)
    rank == 0 && @info("Solution ($(round(time() - t1, digits=3)) [s])")

    if rank == 0
        t1 = time()
        Error = 0.0
        for k in axes(fens.xyz, 1)
            Error = Error + abs.(Temp.values[k, 1] - tempf(reshape(fens.xyz[k, :], (1, 2)))[1])
        end
        Error = Error / count(fens)
        @info("Error =$Error ($(round(time() - t1, digits=3)) [s])")
        if visualize
            geometry = xyz3(fens)
            geometry[:, 3] .= Temp.values
            fens.xyz = geometry
            File = "Poisson2D_examples-sol.vtk"
            MeshExportModule.VTK.vtkexportmesh(File, fens, fes; scalars=[("Temp", Temp.values)])
        end
    end

    MPI.Finalize()
    
    true
end # _execute

function test(; kind = "Q8", N = 253, nbf1max = 2, nelperpart = 2*(nbf1max+1)^2, overlap = 1, itmax = 1000, relrestol = 1e-6, visualize = false)
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
    _execute(N, mesher, volrule, nelperpart, nbf1max, overlap, itmax, relrestol, visualize)
end

test()

nothing
end # module Poisson2D_mpi_driver
