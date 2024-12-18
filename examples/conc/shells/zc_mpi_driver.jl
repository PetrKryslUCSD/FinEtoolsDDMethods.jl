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
"""
module zc_mpi_examples

using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsDeforLinear
using FinEtoolsFlexStructures.FESetShellT3Module: FESetShellT3
using FinEtoolsFlexStructures.FESetShellQ4Module: FESetShellQ4
using FinEtoolsFlexStructures.FEMMShellT3FFModule
using FinEtoolsFlexStructures.RotUtilModule: initial_Rfield, update_rotation_field!
using FinEtoolsDDMethods
using FinEtoolsDDMethods.CGModule: pcg
using FinEtoolsDDMethods.PartitionCoNCModule: CoNCPartitioningInfo, npartitions
using FinEtoolsDDMethods.DDCoNCMPIModule: DDCoNCMPIComm, PartitionedVector, aop!, TwoLevelPreConditioner, rhs
using FinEtoolsDDMethods.DDCoNCMPIModule: vec_collect, vec_copyto!
using FinEtoolsDDMethods.CoNCUtilitiesModule: patch_coordinates
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
using ShellStructureTopo: create_partitions
using MPI

# using MatrixSpy

function zcant!(csmatout, XYZ, tangents, feid, qpid)
    r = vec(XYZ); 
    cross3!(r, view(tangents, :, 1), view(tangents, :, 2))
    csmatout[:, 3] .= vec(r)/norm(vec(r))
    csmatout[:, 1] .= (1.0, 0.0, 0.0)
    cross3!(view(csmatout, :, 2), view(csmatout, :, 3), view(csmatout, :, 1))
    return csmatout
end



function _execute_alt(filename, ref, Nc, n1, Np, No, itmax, relrestol, peek, visualize)
    # Parameters:
    E = 210e9
    nu = 0.3
    L = 10.0
    CTE = 0.0
    thickness = 0.1
    tolerance = thickness / 1000
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
        10 1 1
    ]
    
    to = time()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    @assert Np == MPI.Comm_size(comm)
    Np = MPI.Comm_size(comm)
    # rank == 0 && (@info "$(MPI.versioninfo())")
    rank == 0 && (@info "$(MPI.Get_library_version())")

    BLAS_THREADS = parse(Int, """$(get(ENV, "BLAS_THREADS", 1))""")
    rank == 0 && (@info "BLAS_THREADS = $(BLAS_THREADS)")
    BLAS.set_num_threads(BLAS_THREADS)

    rank == 0 && (@info "Number of processes/partitions: $Np")

    t1 = time()
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
    
    bfes = meshboundary(fes)
    el = selectelem(fens, bfes, box = Float64[10.0 10.0 1.0 1.0 -1 1], inflate = tolerance)
    lfemm1 = FEMMBase(IntegDomain(subset(bfes, el), GaussRule(1, 2)))
    fi1 = ForceIntensity(Float64[0, 0, +0.6e6, 0, 0, 0]);
    el = selectelem(fens, bfes, box = Float64[10.0 10.0 -1.0 -1.0 -1 1], inflate = tolerance)
    lfemm2 = FEMMBase(IntegDomain(subset(bfes, el), GaussRule(1, 2)))
    fi2 = ForceIntensity(Float64[0, 0, -0.6e6, 0, 0, 0]);
    F = distribloads(lfemm1, geom0, dchi, fi1, 1) + distribloads(lfemm2, geom0, dchi, fi2, 1);
    F_f = F[fr]

    associategeometry!(femm, geom0)

    rank == 0 && (@info("Refinement factor: $(ref)"))
    rank == 0 && (@info("Number of overlaps: $(No)"))
    rank == 0 && (@info("Number of nodes: $(count(fens))"))
    rank == 0 && (@info("Number of elements: $(count(fes))"))
    rank == 0 && (@info("Number of free dofs = $(nfreedofs(dchi))"))

    rank == 0 && (@info("Startup ($(round(time() - t1, digits=3)) [s])"))

    function make_matrix(fes)
        femm1 = deepcopy(femm) # for thread safety
        femm1.integdomain.fes = fes
        return stiffness(femm1, geom0, u0, Rfield0, dchi);
    end

    t1 = time()
    cpi = CoNCPartitioningInfo(fens, fes, Np, No, dchi) 
    rank == 0 && (@info("Create partitioning info ($(round(time() - t1, digits=3)) [s])"))
    t2 = time()
    ddcomm = DDCoNCMPIComm(comm, cpi, fes, make_matrix, nothing)
    rank == 0 && (@info("Make partitions ($(round(time() - t2, digits=3)) [s])"))
    meanps = mean_partition_size(cpi)
    rank == 0 && (@info("Mean fine partition size: $(meanps)"))
    rank == 0 && (@info("Create partitions ($(round(time() - t1, digits=3)) [s])"))

    t1 = time()
    rank == 0 && (@info("Number of clusters (requested): $(Nc)"))
    rank == 0 && (@info("Number of 1D basis functions: $(n1)"))
    nt = n1*(n1+1)/2 
    (Nc == 0) && (Nc = Int(floor(meanps / nt / ndofs(dchi))))
    Nepc = count(fes) ÷ Nc
    (n1 > (Nepc/2)^(1/2)) && @error "Not enough elements per cluster"
    rank == 0 && (@info("Number of elements per cluster: $(Nepc)"))
    
    cpartitioning, Nc = shell_cluster_partitioning(fens, fes, Nepc)
    rank == 0 && (@info("Number of clusters (actual): $(Nc)"))
        
    mor = CoNCData(list -> patch_coordinates(fens.xyz, list), cpartitioning)
    Phi = transfmatrix(mor, LegendreBasis, n1, dchi)
    Phi = Phi[fr, :]
    rank == 0 && (@info("Size of the reduced problem: $(size(Phi, 2))"))
    rank == 0 && (@info("Generate clusters ($(round(time() - t1, digits=3)) [s])"))

    function peeksolution(iter, x, resnorm)
        rank == 0 && peek && (@info("it $(iter): residual norm =  $(resnorm)"))
    end
        
    t1 = time()
    M! = TwoLevelPreConditioner(ddcomm, Phi)
    rank == 0 && (@info("Create preconditioner ($(round(time() - t1, digits=3)) [s])"))

    t0 = time()
    x0 = PartitionedVector(Float64, ddcomm)
    vec_copyto!(x0, 0.0)
    b = PartitionedVector(Float64, ddcomm)
    vec_copyto!(b, F_f)
    (sol, stats) = pcg(
        (q, p) -> aop!(q, p), 
        b, x0;
        (M!)=(q, p) -> M!(q, p),
        peeksolution=peeksolution,
        itmax=itmax, 
        atol= 0, rtol=relrestol, normtype = KSP_NORM_UNPRECONDITIONED
        )
    t1 = time()
    rank == 0 && (@info("Number of iterations:  $(stats.niter)"))
    rank == 0 && (@info("Iterations ($(round(t1 - t0, digits=3)) [s])"))
    stats = (niter = stats.niter, residuals = stats.residuals ./ norm(F_f))
    if rank == 0
        data = Dict(
            "number_nodes" => count(fens),
            "number_elements" => count(fes),
            "nfreedofs" => nfreedofs(dchi),
            "Nc" => Nc,
            "n1" => n1,
            "Np" => Np,
            "No" => No,
            "meanps" => meanps,
            "size_Kr_ff" => size(M!.Kr_ff_factor),
            "stats" => stats,
            "iteration_time" => t1 - t0,
        )
        f = (filename == "" ?
             "zc-" *
             "-ref=$(ref)" *
             "-Nc=$(Nc)" *
             "-n1=$(n1)" *
             "-Np=$(Np)" *
             "-No=$(No)" :
             filename)
        @info("Storing data in $(f * ".json")")
        DataDrop.store_json(f * ".json", data)
    end
    dchi_f = vec_collect(sol)
    scattersysvec!(dchi, dchi_f, DOF_KIND_FREE)

    if rank == 0
        if visualize
            f = (filename == "" ?
                 "zc-" *
                 "-ref=$(ref)" *
                 "-Nc=$(Nc)" *
                 "-n1=$(n1)" *
                 "-Np=$(Np)" *
                 "-No=$(No)" :
                 filename) * "-cg-sol"
            VTK.vtkexportmesh(f * ".vtk", fens, fes;
                vectors=[("u", deepcopy(dchi.values[:, 1:3]),)])
        end
    end

    MPI.Finalize()
    rank == 0 && (@info("Total time: $(round(time() - to, digits=3)) [s]"))
    
    true
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
        default = 5
        "--Np"
        help = "Number of partitions"
        arg_type = Int
        default = 7
        "--ref"
        help = "Refinement factor"
        arg_type = Int
        default = 7
        "--Nepp"
        help = "Number of elements per partition"
        arg_type = Int
        default = 0
        "--itmax"
        help = "Maximum number of iterations allowed"
        arg_type = Int
        default = 200
        "--relrestol"
        help = "Relative residual tolerance"
        arg_type = Float64
        default = 1.0e-6
        "--peek"
        help = "Peek at the iterations?"
        arg_type = Bool
        default = true
        "--visualize"
        help = "Write out visualization files?"
        arg_type = Bool
        default = false
    end
    return parse_args(s)
end

p = parse_commandline()

ref = p["ref"]
Nepp = p["Nepp"]    
Np = p["Np"]
if Nepp == 0 && ref == 0
    error("Either ref or Nepp must be specified")
end
if ref == 0
    ref = Int(ceil(sqrt((Np * Nepp / 48))))
end

_execute_alt(
    p["filename"],
    ref,
    p["Nc"], 
    p["n1"],
    Np,
    p["No"], 
    p["itmax"], 
    p["relrestol"],
    p["peek"],
    p["visualize"]
    )


nothing
end # module


