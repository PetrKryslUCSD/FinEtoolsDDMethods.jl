"""
Free vibration of steel barrel

The structure represents a barrel container with spherical caps, two T
stiffeners, and edge stiffener. This example tests the ability to represent
creases in the surface, and junctions with more than two shell faces joined
along a an intersection.

"""
module barrel_seq_examples

using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsDeforLinear
using FinEtoolsFlexStructures.FESetShellT3Module: FESetShellT3
using FinEtoolsFlexStructures.FESetShellQ4Module: FESetShellQ4
using FinEtoolsFlexStructures.FEMMShellT3FFModule
using FinEtoolsFlexStructures.RotUtilModule: initial_Rfield, update_rotation_field!
using FinEtoolsDDMethods
using FinEtoolsDDMethods.CGModule: pcg
using FinEtoolsDDMethods.CoNCUtilitiesModule: patch_coordinates
using FinEtoolsDDMethods.PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData, npartitions
using FinEtoolsDDMethods.DDCoNCSeqModule: DDCoNCSeqComm, PartitionedVector, aop!, TwoLevelPreConditioner
using FinEtoolsDDMethods.DDCoNCSeqModule:  vec_copyto!, vec_collect
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
using ShellStructureTopo: make_topo_faces, create_partitions

# using MatrixSpy

function _execute_alt(filename, ref, stabilize, Nc, n1, Np, No, itmax, relrestol, peek, visualize)
    # Parameters:
    E = 200e3 * phun("MPa")
    nu = 0.3
    rho = 7850 * phun("KG/M^3")
    thickness = 2.0 * phun("mm")
    pressure = 100.0 * phun("kilo*Pa")
    CTE = 0.0
    input = "barrel_w_stiffeners-s3.h5mesh"

    function computetrac!(forceout, XYZ, tangents, feid, qpid)
   		n = cross(tangents[:, 1], tangents[:, 2]) 
        n = n / norm(n)
        forceout[1:3] = n * pressure
	    forceout[4:6] .= 0.0
	    return forceout
    end

    to = time()

    BLAS_THREADS = parse(Int, """$(get(ENV, "BLAS_THREADS", 1))""")
    rank == 0 && (@info "BLAS_THREADS = $(BLAS_THREADS)")
    BLAS.set_num_threads(BLAS_THREADS)

        
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
    
    lfemm = FEMMBase(IntegDomain(subset(fes, vessel), TriRule(3)))
    fi = ForceIntensity(Float64, 6, computetrac!);
    F = distribloads(lfemm, geom0, dchi, fi, 2);
    F_f = F[fr]

    associategeometry!(femm, geom0)
    
    @info("Refinement factor: $(ref)")
    @info("Number of fine grid partitions: $(Np)")
    @info("Number of overlaps: $(No)")
    @info("Number of nodes: $(count(fens))")
    @info("Number of elements: $(count(fes))")
    @info("Number of free dofs = $(nfreedofs(dchi))")

    function make_matrix(fes)
        femm1 = deepcopy(femm) # for thread safety
        femm1.integdomain.fes = fes
        return stiffness(femm1, geom0, u0, Rfield0, dchi);
    end

    t1 = time()
    cpi = CoNCPartitioningInfo(fens, fes, Np, No, dchi) 
    @info("Create partitioning info ($(round(time() - t1, digits=3)) [s])")
    t2 = time()
    ddcomm = DDCoNCSeqComm(nothing, cpi, fes, make_matrix, nothing)
    @info("Make partitions ($(round(time() - t2, digits=3)) [s])")
    meanps = mean_partition_size(cpi)
    @info("Mean fine partition size: $(meanps)")
    @info("Create partitions ($(round(time() - t1, digits=3)) [s])")

    t1 = time()
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
    @info("Generate clusters ($(round(time() - t1, digits=3)) [s])")

    function peeksolution(iter, x, resnorm)
        peek && (@info("it $(iter): residual norm =  $(resnorm)"))
    end
    
    
    t1 = time()
    M! = TwoLevelPreConditioner(ddcomm, Phi)
    @info("Create preconditioner ($(round(time() - t1, digits=3)) [s])")

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
    @info("Number of iterations:  $(stats.niter)")
    @info("Iterations ($(round(t1 - t0, digits=3)) [s])")
    stats = (niter = stats.niter, residuals = stats.residuals ./ norm(F_f))
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
         "barrel-" *
         "-ref=$(ref)" *
         "-Nc=$(Nc)" *
         "-n1=$(n1)" *
         "-Np=$(Np)" *
         "-No=$(No)" :
         filename)
    @info("Storing data in $(f * ".json")")
    DataDrop.store_json(f * ".json", data)
    dchi_f = vec_collect(sol)
    scattersysvec!(dchi, dchi_f, DOF_KIND_FREE)
    
    if visualize
        f = (filename == "" ?
         "barrel-" *
         "-ref=$(ref)" *
         "-Nc=$(Nc)" *
         "-n1=$(n1)" *
         "-Np=$(Np)" *
         "-No=$(No)" :
         filename) * "-cg-sol"
        VTK.vtkexportmesh(f * ".vtk", fens, fes;
            vectors=[("u", deepcopy(dchi.values[:, 1:3]),)])
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
        "--stabilize"
        help = "Stabilize rotation about axis?"
        arg_type = Bool
        default = true
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
        default = 1
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
    p["ref"],
    p["stabilize"],
    p["Nc"], 
    p["n1"],
    p["Np"], 
    p["No"], 
    p["itmax"], 
    p["relrestol"],
    p["peek"],
    p["visualize"],
    )


nothing
end # module


