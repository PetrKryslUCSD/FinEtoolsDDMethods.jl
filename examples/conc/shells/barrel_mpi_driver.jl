"""
Free vibration of steel barrel

The structure represents a barrel container with spherical caps, two T
stiffeners, and edge stiffener. This example tests the ability to represent
creases in the surface, and junctions with more than two shell faces joined
along a an intersection.

Parallel simulation with MPI.
"""
module barrel_mpi_driver
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

# Parameters:
E = 200e3 * phun("MPa")
nu = 0.3
rho = 7850 * phun("KG/M^3")
thickness = 2.0 * phun("mm")
pressure = 100.0 * phun("kilo*Pa")

input = "barrel_w_stiffeners-s3.h5mesh"
# input = "barrel_w_stiffeners-100.inp"

function computetrac!(forceout, XYZ, tangents, feid, qpid)
    n = cross(tangents[:, 1], tangents[:, 2]) 
    n = n/norm(n)
    forceout[1:3] = n*pressure
    forceout[4:6] .= 0.0
    return forceout
end

function _execute(ncoarse, nelperpart, nbf1max, nfpartitions, overlap, ref, itmax, relrestol, stabilize, visualize)
    CTE = 0.0
        
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    rank == 0 && (@info "$(MPI.Get_library_version())")

    BLAS_THREADS = parse(Int, get(ENV, "BLAS_THREADS", 1))
    rank == 0 && (@info "BLAS_THREADS = $(BLAS_THREADS)")
    BLAS.set_num_threads(BLAS_THREADS)

    nfpartitions = nprocs - 1

    rank == 0 && (@info "Number of processes: $nprocs")
    rank == 0 && (@info "Number of partitions: $nfpartitions")

    if !isfile(joinpath(dirname(@__FILE__()), input))
        success(run(`unzip -qq -d $(dirname(@__FILE__())) $(joinpath(dirname(@__FILE__()), "barrel_w_stiffeners-s3-mesh.zip"))`; wait = false))
    end
    output = FinEtools.MeshImportModule.import_H5MESH(joinpath(dirname(@__FILE__()), input))
    fens, fes  = output["fens"], output["fesets"][1]
    fens.xyz .*= phun("mm");

    box = boundingbox(fens.xyz)
    middle = [(box[2] + box[1])/2, (box[4] + box[3])/2, (box[6] + box[5])/2]
    fens.xyz[:, 1] .-= middle[1]
    fens.xyz[:, 2] .-= middle[2]
    fens.xyz[:, 3] .-= middle[3]

    # output = import_ABAQUS(joinpath(dirname(@__FILE__()), input))
    # fens = output["fens"]
    # fes = output["fesets"][1]

    connected = findunconnnodes(fens, fes);
    fens, new_numbering = compactnodes(fens, connected);
    fes = renumberconn!(fes, new_numbering);

    fens, fes = make_topo_faces(fens, fes)
    unique_surfaces =  unique(fes.label)
    # for i in 1:length(unique_surfaces)
    #     el = selectelem(fens, fes, label = unique_surfaces[i])
    #     VTKWrite.vtkwrite("surface-$(unique_surfaces[i]).vtk", fens, subset(fes, el))
    # end
    # return
    vessel = []
    for i in [1, 2, 6, 7, 8]
        el = selectelem(fens, fes, label = unique_surfaces[i])
        append!(vessel, el)
    end

    # for r in 1:ref
    #     fens, fes = T3refine(fens, fes)
    # end
    
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
    l1 = selectnode(fens; box = Float64[0 0 -Inf Inf 0 0], inflate = 0.01)
    for i in [1, 3, 4, 5, 6]
        setebc!(dchi, [l1[1], l1[end]], true, i)
    end
    l1 = selectnode(fens; box = Float64[-Inf Inf 0 0 -Inf Inf], inflate = 0.01)
    for i in [2]
        setebc!(dchi, l1[1:1], true, i)
    end
    if stabilize
        l1 = selectnode(fens; box = Float64[0 0 0 0 -Inf Inf], inflate = 0.02)
        for i in [1, ]
            setebc!(dchi, l1[1:1], true, i)
        end
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

    lfemm = FEMMBase(IntegDomain(subset(fes, vessel), TriRule(3)))
    fi = ForceIntensity(Float64, 6, computetrac!);
    F = distribloads(lfemm, geom0, dchi, fi, 2);
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
        @info "DEBUG rank=$rank, $(length(partition.odof)) ($(round(time() - t1, digits=3)) [s])"
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

function test(;nelperpart = 200, nbf1max = 3, nfpartitions = 8, overlap = 3, ref = 1, stabilize = false, itmax = 2000, relrestol = 1e-6, visualize = false) 
    _execute(32, nelperpart, nbf1max, nfpartitions, overlap, ref, itmax, relrestol, stabilize, visualize)
end

test()

nothing
end # module


