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
module LE5_Z_cantilever_mpi_driver
using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsDeforLinear
using FinEtoolsFlexStructures.FESetShellT3Module: FESetShellT3
using FinEtoolsFlexStructures.FESetShellQ4Module: FESetShellQ4
using FinEtoolsFlexStructures.FEMMShellT3FFModule
using FinEtoolsFlexStructures.RotUtilModule: initial_Rfield, update_rotation_field!
using MPI
using FinEtoolsDDMethods
using FinEtoolsDDMethods.CGModule: pcg_mpi_2level_Schwarz
using FinEtoolsDDMethods.PartitionCoNCDDMPIModule
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

# using MatrixSpy

function zcant!(csmatout, XYZ, tangents, feid, qpid)
    r = vec(XYZ); 
    cross3!(r, view(tangents, :, 1), view(tangents, :, 2))
    csmatout[:, 3] .= vec(r)/norm(vec(r))
    csmatout[:, 1] .= (1.0, 0.0, 0.0)
    cross3!(view(csmatout, :, 2), view(csmatout, :, 3), view(csmatout, :, 1))
    return csmatout
end

# Parameters:
E = 210e9
nu = 0.3
L = 10.0
input = "nle5xf3c.inp"

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

function _execute(ncoarse, aspect, nelperpart, nbf1max, overlap, ref, itmax, relrestol, visualize)
    CTE = 0.0
    distortion = 0.0
    n = ncoarse  * ref    # number of elements along the edge of the block
    thickness = 0.1
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    nfpartitions = nprocs - 1

    rank == 0 && (@info "Number of processes: $nprocs")
    rank == 0 && (@info "Number of partitions: $nfpartitions")

    tolerance = thickness/1000
    output = import_ABAQUS(joinpath(dirname(@__FILE__()), input))
    fens = output["fens"]
    fes = output["fesets"][1]

    connected = findunconnnodes(fens, fes);
    fens, new_numbering = compactnodes(fens, connected);
    fes = renumberconn!(fes, new_numbering);

    for r in 1:ref
        fens, fes = T3refine(fens, fes)
    end
    
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
    rank == 0 && (@info("Number of elements per partition: $(nelperpart)"))
    rank == 0 && (@info("Number of 1D basis functions: $(nbf1max)"))
    rank == 0 && (@info("Number of fine grid partitions: $(nfpartitions)"))
    rank == 0 && (@info("Overlap: $(overlap)"))
    rank == 0 && (@info("Number of elements: $(count(fes))"))
    rank == 0 && (@info("Number of free dofs = $(nfreedofs(dchi))"))

    nl = selectnode(fens; box = Float64[10.0 10.0 1.0 1.0 0 0], tolerance = tolerance)
    loadbdry1 = FESetP1(reshape(nl, 1, 1))
    lfemm1 = FEMMBase(IntegDomain(loadbdry1, PointRule()))
    fi1 = ForceIntensity(Float64[0, 0, +0.6e6, 0, 0, 0]);
    nl = selectnode(fens; box = Float64[10.0 10.0 -1.0 -1.0 0 0], tolerance = tolerance)
    loadbdry2 = FESetP1(reshape(nl, 1, 1))
    lfemm2 = FEMMBase(IntegDomain(loadbdry2, PointRule()))
    fi2 = ForceIntensity(Float64[0, 0, -0.6e6, 0, 0, 0]);
    F = distribloads(lfemm1, geom0, dchi, fi1, 3) + distribloads(lfemm2, geom0, dchi, fi2, 3);
    F_f = F[fr]

    associategeometry!(femm, geom0)
   
    t1 = time()
    cpartitioning, ncpartitions = shell_cluster_partitioning(fens, fes, nelperpart)
    rank == 0 && (@info("Number of clusters (coarse grid partitions): $(ncpartitions)"))
        
    mor = CoNCData(list -> patch_coordinates(fens.xyz, list), cpartitioning)
    Phi = transfmatrix(mor, LegendreBasis, nbf1max, dchi)
    Phi = Phi[fr, :]
    rank == 0 && (@info("Size of the reduced problem: $(size(Phi, 2))"))
    rank == 0 && (@info "Generate clusters ($(round(time() - t1, digits=3)) [s])")

    function make_matrix(fes)
        femm.integdomain.fes = fes
        return stiffness(femm, geom0, u0, Rfield0, dchi);
    end

    t1 = time()
    partition = nothing
    if rank > 0
        cpi = CoNCPartitioningInfo(fens, fes, nfpartitions, overlap, dchi) 
        partition = CoNCPartitionData(cpi, rank, fes, Phi, make_matrix, nothing)
    end    
    MPI.Barrier(comm)
    rank == 0 && (@info "Create partitions ($(round(time() - t1, digits=3)) [s])")

    function peeksolution(iter, x, resnorm)
        @info("Iteration $(iter): residual norm $(resnorm)")
    end
    
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
    rank == 0 && (@info "Create global factor ($(round(time() - t1, digits=3)) [s])")
    
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
    scattersysvec!(dchi, u_f, DOF_KIND_FREE)
    rank == 0 && @info("Solution ($(round(time() - t1, digits=3)) [s])")
    
    MPI.Finalize()
    
    true
end

function test(;aspect = 100, nelperpart = 200, nbf1max = 2, overlap = 1, ref = 1, itmax = 2000, relrestol = 1e-6, visualize = false) 
    _execute(32, aspect, nelperpart, nbf1max, overlap, ref, itmax, relrestol, visualize)
end

test()

nothing
end # module


