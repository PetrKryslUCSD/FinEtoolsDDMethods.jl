"""
MODEL DESCRIPTION

Z-section cantilever under torsional loading.

Linear elastic analysis, Young's modulus = 210 GPa, Poisson's ratio = 0.3.

All displacements are fixed at X=0.

Torque of 1.2 MN-m applied at X=10. The torque is applied by two 
uniformly distributed shear loads of 0.6 MN at each flange surface.

Objective of the analysis is to compute the axial stress at X = 2.5 from fixed end.

NAFEMS REFERENCE SOLUTION

Axial stress at X = 2.5 from fixed end (point A) at the midsurface is -108 MPa.

Parallel simulation with MPI.
"""
module zc_mpi_driver
using FinEtools
using FinEtools.MeshExportModule: VTK, VTKWrite
using FinEtoolsDeforLinear
using FinEtoolsFlexStructures.FESetShellT3Module: FESetShellT3
using FinEtoolsFlexStructures.FESetShellQ4Module: FESetShellQ4
using FinEtoolsFlexStructures.FEMMShellT3FFModule
using FinEtoolsFlexStructures.RotUtilModule: initial_Rfield, update_rotation_field!
using MPI
using FinEtoolsDDMethods
using FinEtoolsDDMethods.CGModule: pcg_mpi_2level_Schwarz
using FinEtoolsDDMethods.DDCoNCMPIModule: partition_mult!, precond_2level!, MPIAOperator, MPITwoLevelPreconditioner
using FinEtoolsDDMethods.CoNCUtilitiesModule: patch_coordinates, conc_cache
using FinEtoolsDDMethods: set_up_timers
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
using ILUZero
using Statistics
using ShellStructureTopo: make_topo_faces, create_partitions


# using MatrixSpy

function zcant!(csmatout, XYZ, tangents, feid, qpid)
    r = vec(XYZ); 
    cross3!(r, view(tangents, :, 1), view(tangents, :, 2))
    csmatout[:, 3] .= vec(r)/norm(vec(r))
    csmatout[:, 1] .= (1.0, 0.0, 0.0)
    cross3!(view(csmatout, :, 2), view(csmatout, :, 3), view(csmatout, :, 1))
    return csmatout
end

function _execute(filename, ref, Nc, n1, No, itmax, relrestol, peek, visualize)
    E = 210e9
    nu = 0.3
    L = 10.0
    CTE = 0.0
    thickness = 0.1
    tolerance = thickness/1000
    xyz = Float64[
    0 -1 -1;
    2.5 -1 -1;
    10 -1 -1;
    0 -1 0;
    2.5 -1 0;
    10 -1 0;
    0 1 0;
    2.5 1 0;
    10 1 0;
    0 1 1;
    2.5 1 1;
    10 1 1;
    ]

    to = time()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    rank == 0 && (@info "$(MPI.Get_library_version())")

    BLAS_THREADS = parse(Int, """$(get(ENV, "BLAS_THREADS", 1))""")
    rank == 0 && (@info "BLAS_THREADS = $(BLAS_THREADS)")
    BLAS.set_num_threads(BLAS_THREADS)

    Np = nprocs - 1

    rank == 0 && (@info "Number of processes: $nprocs")
    rank == 0 && (@info "Number of partitions: $Np")

    fens1, fes1 = Q4quadrilateral(xyz[[1, 2, 5, 4], :], ref * 2, ref * 1) 
    fens2, fes2 = Q4quadrilateral(xyz[[4, 5, 8, 7], :], ref * 2, ref * 1) 
    fens, fes1, fes2 = mergemeshes(fens1, fes1,  fens2, fes2, tolerance)
    fes = cat(fes1, fes2)

    fens1, fes1 = fens, fes 
    fens2, fes2 = Q4quadrilateral(xyz[[7, 8, 11, 10], :], ref * 2, ref * 1) 
    fens, fes1, fes2 = mergemeshes(fens1, fes1,  fens2, fes2, tolerance)
    fes = cat(fes1, fes2)

    fens1, fes1 = fens, fes 
    fens2, fes2 = Q4quadrilateral(xyz[[2, 3, 6, 5], :], ref * 6, ref * 1) 
    fens, fes1, fes2 = mergemeshes(fens1, fes1,  fens2, fes2, tolerance)
    fes = cat(fes1, fes2)

    fens1, fes1 = fens, fes 
    fens2, fes2 = Q4quadrilateral(xyz[[5, 6, 9, 8], :], ref * 6, ref * 1) 
    fens, fes1, fes2 = mergemeshes(fens1, fes1,  fens2, fes2, tolerance)
    fes = cat(fes1, fes2)

    fens1, fes1 = fens, fes 
    fens2, fes2 = Q4quadrilateral(xyz[[8, 9, 12, 11], :], ref * 6, ref * 1) 
    fens, fes1, fes2 = mergemeshes(fens1, fes1,  fens2, fes2, tolerance)
    fes = cat(fes1, fes2)

    fens, fes = Q4toT3(fens, fes)
    
    MR = DeforModelRed3D
    mater = MatDeforElastIso(MR, 0.0, E, nu, CTE)
    ocsys = CSys(3, 3, zcant!)

    sfes = FESetShellT3()
    accepttodelegate(fes, sfes)
    femm = FEMMShellT3FFModule.make(IntegDomain(fes, TriRule(1), thickness), mater)
    stiffness = FEMMShellT3FFModule.stiffness
    associategeometry! = FEMMShellT3FFModule.associategeometry!

    # Construct the requisite fields, geometry and displacement
    # Initialize configuration variables
    geom0 = NodalField(fens.xyz)
    u0 = NodalField(zeros(size(fens.xyz,1), 3))
    Rfield0 = initial_Rfield(fens)
    dchi = NodalField(zeros(size(fens.xyz,1), 6))

    # Apply EBC's
    # plane of symmetry perpendicular to Z
    l1 = selectnode(fens; box = Float64[0 0 -Inf Inf -Inf Inf], inflate = tolerance)
    for i in [1,2,3,]
        setebc!(dchi, l1, true, i)
    end
    applyebc!(dchi)
    numberdofs!(dchi);

    fr = dofrange(dchi, DOF_KIND_FREE)
    dr = dofrange(dchi, DOF_KIND_DATA)
    
    rank == 0 && (@info("Refinement factor: $(ref)"))
    rank == 0 && (@info("Number of fine grid partitions: $(Np)"))
    rank == 0 && (@info("Number of overlaps: $(No)"))
    rank == 0 && (@info("Number of elements: $(count(fes))"))
    rank == 0 && (@info("Number of free dofs = $(nfreedofs(dchi))"))

    bfes = meshboundary(fes)
    el = selectelem(fens, bfes, box = Float64[10.0 10.0 1.0 1.0 -1 1], tolerance = tolerance)
    lfemm1 = FEMMBase(IntegDomain(subset(bfes, el), GaussRule(1, 2)))
    fi1 = ForceIntensity(Float64[0, 0, +0.6e6, 0, 0, 0]);
    el = selectelem(fens, bfes, box = Float64[10.0 10.0 -1.0 -1.0 -1 1], tolerance = tolerance)
    lfemm2 = FEMMBase(IntegDomain(subset(bfes, el), GaussRule(1, 2)))
    fi2 = ForceIntensity(Float64[0, 0, -0.6e6, 0, 0, 0]);
    F = distribloads(lfemm1, geom0, dchi, fi1, 1) + distribloads(lfemm2, geom0, dchi, fi2, 1);
    F_f = F[fr]

    associategeometry!(femm, geom0)
          
    function make_matrix(fes)
        femm1 = deepcopy(femm) # for thread safety
        femm1.integdomain.fes = fes
        return stiffness(femm1, geom0, u0, Rfield0, dchi);
    end

    t1 = time()
    cpi = nothing
    if rank == 0
        cpi = CoNCPartitioningInfo(fens, fes, Np, No, dchi) 
    end
    cpi = MPI.bcast(cpi, 0, comm)
    rank == 0 && (@info "Create partitioning info ($(round(time() - t1, digits=3)) [s])")
    rank == 0 && (@info "Mean partition size: $(mean_partition_size(cpi))")
    
    t1 = time()
    partition = nothing
    Phi = nothing
    if rank > 0
        partition = CoNCPartitionData(cpi, rank, fes, make_matrix, nothing)
    else
        @info("Number of clusters (requested): $(Nc)")
        @info("Number of 1D basis functions: $(n1)")
        nt = n1*(n1+1)/2 
        (Nc == 0) && (Nc = Int(floor(meanps / nt / ndofs(dchi))))
        Nepc = count(fes) ÷ Nc
        (n1 > (Nepc/2)^(1/2)) && @error "Not enough elements per cluster"
        @info("Number of elements per cluster: $(Nepc)")
    
        cpartitioning, Nc = shell_cluster_partitioning(fens, fes, Nepc)
        @info("Number of clusters (actual): $(Nc)")
        
        mor = CoNCData(list -> patch_coordinates(fens.xyz, list), cpartitioning)
        Phi = transfmatrix(mor, LegendreBasis, n1, dchi)
        Phi = Phi[fr, :]
        @info("Size of the reduced problem: $(size(Phi, 2))")
        @info "Generate clusters ($(round(time() - t1, digits=3)) [s])"
    end    
    MPI.Barrier(comm)
    Phi = MPI.bcast(Phi, 0, comm)
    rank == 0 && (@info "Create partitions and clusters ($(round(time() - t1, digits=3)) [s])")

    t1 = time()
    Kr_ff = spzeros(size(Phi, 2), size(Phi, 2))
    if rank > 0
        Kr_ff = (Phi' * partition.nonoverlapping_K * Phi)
    end
    Krfactor = nothing
    ks = MPI.gather(Kr_ff, comm; root=0)
    if rank == 0
        for k in ks
            Kr_ff += k
        end
        Krfactor = lu(Kr_ff)
    end
    ccache = conc_cache(Krfactor, Phi)
    rank == 0 && (@info "Create global factor ($(round(time() - t1, digits=3)) [s])")
    
    function peeksolution(iter, x, resnorm)
        (rank == 0 && peek) && (@info("it $(iter): residual norm =  $(resnorm)"))
    end
    
    t1 = time()
    norm_F_f = norm(F_f) 
    aop = MPIAOperator(comm, rank, partition, cpi,
        (rank == 0
         ? set_up_timers("1_send_nbuffs", "2_add_nbuffs", "3_total")
         : set_up_timers("1_mult_local"))
    )
    pre = MPITwoLevelPreconditioner(comm, rank, partition, cpi, ccache, 
        (rank == 0 
        ? set_up_timers("1_send_obuffs", "2_solve_global", "3_wait_obuffs", "4_add_obuffs", "5_total")
        : set_up_timers("1_solve_local"))
    )
    (u_f, stats) = pcg_mpi_2level_Schwarz(
        comm, 
        rank,
        (q, p) -> partition_mult!(q, aop, p),
        F_f,
        zeros(size(F_f)),
        (q, p) -> precond_2level!(q, pre, p);
        itmax=itmax, atol=0.0, rtol=relrestol, normtype = KSP_NORM_UNPRECONDITIONED,
        peeksolution=peeksolution)
    rank == 0 && (@info("Number of iterations:  $(stats.niter)"))
    rank == 0 && (@info "Iterations ($(round(time() - t1, digits=3)) [s])")
    stats = (niter = stats.niter, residuals = stats.residuals ./ norm(F_f), timers = stats.timers)

    if rank > 0
        MPI.send(stats.timers, comm; dest = 0)
    end
    root_timers = stats.timers
    pavg_timers = Dict([(k, 0.0) for k in keys(root_timers)])
    if rank == 0
        ptimers = [MPI.recv(comm; source=i) for i in 1:Np]
        for k in keys(root_timers)
            root_timers[k] = root_timers[k] / stats.niter
            pavg_timers[k] = mean([t[k] for t in ptimers]) / stats.niter 
        end
    end

    if rank > 0
        MPI.send(aop.timers, comm; dest = 0)
    end
    aop_pavg_timers = nothing
    if rank == 0
        ptimers = [MPI.recv(comm; source=i) for i in 1:Np]
        aop_pavg_timers = Dict([(k, 0.0) for k in keys(ptimers[1])])
        for k in keys(ptimers[1])
            aop_pavg_timers[k] = mean([t[k] for t in ptimers]) / stats.niter
        end
        for k in keys(aop.timers)
            aop.timers[k] = aop.timers[k] / stats.niter
        end
    end

    if rank > 0
        MPI.send(pre.timers, comm; dest = 0)
    end
    pre_pavg_timers = nothing
    if rank == 0
        ptimers = [MPI.recv(comm; source=i) for i in 1:Np]
        pre_pavg_timers = Dict([(k, 0.0) for k in keys(ptimers[1])])
        for k in keys(ptimers[1])
            pre_pavg_timers[k] = mean([t[k] for t in ptimers]) / stats.niter
        end
        for k in keys(pre.timers)
            pre.timers[k] = pre.timers[k] / stats.niter
        end
    end

    if rank == 0
        sorted(k) = sort(collect(keys(k)))
        data = Dict(
            "number_nodes" => count(fens),
            "number_elements" => count(fes),
            "nfreedofs" => nfreedofs(dchi),
            "Nc" => Nc,
            "n1" => n1,
            "Np" => Np,
            "No" => No,
            "size_Kr_ff" => size(Krfactor),
            "stats" => stats,
            "iteration_time" => time() - t1,
            "timers" => [
                ["aop_root", [[s[1], s[2]] for s in sort(collect(aop.timers), by = x -> x[1])]],
                ["aop_pavg", [[s[1], s[2]] for s in sort(collect(aop_pavg_timers), by = x -> x[1])]],
                ["pre_root", [[s[1], s[2]] for s in sort(collect(pre.timers), by = x -> x[1])]],
                ["pre_pavg", [[s[1], s[2]] for s in sort(collect(pre_pavg_timers), by = x -> x[1])]],
                ["ite_root", [[s[1], s[2]] for s in sort(collect(root_timers), by = x -> x[1])]],
                ["ite_pavg", [[s[1], s[2]] for s in sort(collect(pavg_timers), by = x -> x[1])]],
            ]
        )
        f = (filename == "" ?
             "zc-" *
             "-ref=$(ref)" *
             "-Nc=$(Nc)" *
             "-n1=$(n1)" *
             "-Np=$(Np)" *
             "-No=$(No)" :
             filename)
        @info "Storing data in $(f * ".json")"
        DataDrop.store_json(f * ".json", data)
    end
    
    MPI.Finalize()
    rank == 0 && (@info("Total time: $(round(time() - to, digits=3)) [s]"))
    
    true
end

function test(;filename = "", ref = 3, Nc = 2, n1 = 6, No = 1, itmax = 2000, relrestol = 1e-6, peek = false, visualize = false) 
    _execute(filename, ref, Nc, n1, No, itmax, relrestol, peek, visualize)
end


using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--filename"
        help = "Use filename to name the output files"
        arg_type = String
        default = ""
        "--Nc"
        help = "Number of clusters"
        arg_type = Int
        default = 2
        "--n1"
        help = "Number 1D basis functions"
        arg_type = Int
        default = 5
        "--No"
        help = "Number of overlaps"
        arg_type = Int
        default = 1
        "--ref"
        help = "Refinement factor (increment by 1 increases the number of triangles by a factor of 4)"
        arg_type = Int
        default = 5
        "--itmax"
        help = "Maximum number of iterations allowed"
        arg_type = Int
        default = 2000
        "--relrestol"
        help = "Relative residual tolerance"
        arg_type = Float64
        default = 1.0e-6
        "--peek"
        help = "Peek at the iterations?"
        arg_type = Bool
        default = false
        "--visualize"
        help = "Write out visualization files?"
        arg_type = Bool
        default = false
    end
    return parse_args(s)
end

p = parse_commandline()

_execute(
    p["filename"],
    p["ref"],
    p["Nc"], 
    p["n1"],
    p["No"], 
    p["itmax"], 
    p["relrestol"],
    p["peek"],
    p["visualize"]
    )


nothing
end # module


