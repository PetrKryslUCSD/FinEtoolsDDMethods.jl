
module hyp_mpi_examples

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

function _execute_alt(filename, aspect, ncoarse, ref, Nc, n1, Np, No, itmax, relrestol, peek, visualize)
    # Parameters:
    E = 2.0e11
    nu = 1 / 3
    pressure = 1.0e3
    Length = 2.0
    CTE = 0.0
    distortion = 0.0
    n = ncoarse  * ref    # number of elements along the edge of the block
    tolerance = Length / n/ 100
    thickness = Length/2/aspect

    function computetrac!(forceout, XYZ, tangents, feid, qpid)
        r = vec(XYZ); r[2] = 0.0
        r .= vec(r)/norm(vec(r))
        theta = atan(r[3], r[1])
        n = cross(tangents[:, 1], tangents[:, 2]) 
        n = n/norm(n)
        forceout[1:3] = n*pressure*cos(2*theta)
        forceout[4:6] .= 0.0
        # @show dot(n, forceout[1:3])
        return forceout
    end
    
    to = time()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    Np = MPI.Comm_size(comm)
    rank == 0 && (@info "$(MPI.Get_library_version())")

    BLAS_THREADS = parse(Int, """$(get(ENV, "BLAS_THREADS", 1))""")
    rank == 0 && (@info "BLAS_THREADS = $(BLAS_THREADS)")
    BLAS.set_num_threads(BLAS_THREADS)

    rank == 0 && (@info "Number of processes/partitions: $Np")

    fens, fes = distortblock(T3block, 90/360*2*pi, Length/2, n, n, distortion, distortion);
    fens.xyz = xyz3(fens)
    for i in 1:count(fens)
        a=fens.xyz[i, 1]; y=fens.xyz[i, 2];
        R = sqrt(1 + y^2)
        fens.xyz[i, :] .= (R*sin(a), y, R*cos(a))
    end
    
    MR = DeforModelRed3D
    mater = MatDeforElastIso(MR, 0.0, E, nu, CTE)

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
    l1 = selectnode(fens; box = Float64[-Inf Inf -Inf Inf 0 0], inflate = tolerance)
    for i in [3,4,5]
        setebc!(dchi, l1, true, i)
    end
    # plane of symmetry perpendicular to Y
    l1 = selectnode(fens; box = Float64[-Inf Inf 0 0 -Inf Inf], inflate = tolerance)
    for i in [2,4,6]
        setebc!(dchi, l1, true, i)
    end
    # plane of symmetry perpendicular to X
    l1 = selectnode(fens; box = Float64[0 0 -Inf Inf -Inf Inf], inflate = tolerance)
    for i in [1,5,6]
        setebc!(dchi, l1, true, i)
    end
    # clamped edge perpendicular to Y
    # l1 = selectnode(fens; box = Float64[-Inf Inf L/2 L/2 -Inf Inf], inflate = tolerance)
    # for i in [1,2,3,4,5,6]
    #     setebc!(dchi, l1, true, i)
    # end
    applyebc!(dchi)
    numberdofs!(dchi);

    fr = dofrange(dchi, DOF_KIND_FREE)
    dr = dofrange(dchi, DOF_KIND_DATA)
    
    rank == 0 && (@info("Aspect ratio: $(aspect)"))
    rank == 0 && (@info("Refinement factor: $(ref)"))
    rank == 0 && (@info("Number of fine grid partitions: $(Np)"))
    rank == 0 && (@info("Number of overlaps: $(No)"))
    rank == 0 && (@info("Number of elements: $(count(fes))"))
    rank == 0 && (@info("Number of free dofs = $(nfreedofs(dchi))"))

    lfemm = FEMMBase(IntegDomain(fes, TriRule(3)))
    fi = ForceIntensity(Float64, 6, computetrac!);
    F = distribloads(lfemm, geom0, dchi, fi, 2);
    F_f = F[fr]

    associategeometry!(femm, geom0)
    
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
             "hyp-" *
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
                 "hyp-" *
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
        "--aspect"
        help = "Aspect ratio"
        arg_type = Float64
        default = 10.0
        "--ncoarse"
        help = "Number of edges in the coarse mesh"
        arg_type = Int
        default = 32
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
        "--Np"
        help = "Number of partitions"
        arg_type = Int
        default = 7
        "--ref"
        help = "Refinement factor"
        arg_type = Int
        default = 2
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

_execute_alt(
    p["filename"],
    p["aspect"],
    p["ncoarse"],
    p["ref"],
    p["Nc"], 
    p["n1"],
    p["Np"], 
    p["No"], 
    p["itmax"], 
    p["relrestol"],
    p["peek"],
    p["visualize"]
    )


nothing
end # module


