"""
Free vibration of steel barrel

The structure represents a barrel container with spherical caps, two T
stiffeners, and edge stiffener. This example tests the ability to represent
creases in the surface, and junctions with more than two shell faces joined
along a an intersection.

"""
module barrel_seq_examples
using FinEtools
using FinEtools.MeshExportModule: VTK, VTKWrite
using FinEtoolsDeforLinear
using FinEtoolsFlexStructures.FESetShellT3Module: FESetShellT3
using FinEtoolsFlexStructures.FESetShellQ4Module: FESetShellQ4
using FinEtoolsFlexStructures.FEMMShellT3FFModule
using FinEtoolsFlexStructures.RotUtilModule: initial_Rfield, update_rotation_field!
using FinEtoolsDDMethods
using FinEtoolsDDMethods.CGModule: pcg_seq
using FinEtoolsDDMethods.CoNCUtilitiesModule: patch_coordinates
using FinEtoolsDDMethods.DDCoNCSeqModule: partition_multiply!, preconditioner!, make_partitions
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
    
    @info("Refinement factor: $(ref)")
    @info("Number of elements per partition: $(nelperpart)")
    @info("Number of 1D basis functions: $(nbf1max)")
    @info("Number of fine grid partitions: $(nfpartitions)")
    @info("Overlap: $(overlap)")
    @info("Number of elements: $(count(fes))")
    @info("Number of free dofs = $(nfreedofs(dchi))")

    lfemm = FEMMBase(IntegDomain(subset(fes, vessel), TriRule(3)))
    fi = ForceIntensity(Float64, 6, computetrac!);
    F = distribloads(lfemm, geom0, dchi, fi, 2);
    F_f = F[fr]

    associategeometry!(femm, geom0)
    
    t1 = time()
    cpartitioning, ncpartitions = shell_cluster_partitioning(fens, fes, nelperpart)
    @info("Number of clusters (coarse grid partitions): $(ncpartitions)")
        
    mor = CoNCData(list -> patch_coordinates(fens.xyz, list), cpartitioning)
    Phi = transfmatrix(mor, LegendreBasis, nbf1max, dchi)
    Phi = Phi[fr, :]
    @info("Size of the reduced problem: $(size(Phi, 2))")
    @info("Transformation matrix: $(mebibytes(Phi)) [MiB]")
    @info "Generate clusters ($(round(time() - t1, digits=3)) [s])"

    function make_matrix(fes)
        femm.integdomain.fes = fes
        return stiffness(femm, geom0, u0, Rfield0, dchi);
    end

    t1 = time()
    cpi = CoNCPartitioningInfo(fens, fes, nfpartitions, overlap, dchi) 
    partition_list  = make_partitions(cpi, fes, make_matrix, nothing)
    @info "Mean fine partition size = $(mean([partition_size(_p) for _p in partition_list]))"
    @info "Mean partition allocations: $(mean([mebibytes(_p) for _p in partition_list])) [MiB]" 
    @info "Total partition allocations: $(sum([mebibytes(_p) for _p in partition_list])) [MiB]" 
    @info "Create partitions time: $(time() - t1)"

    function peeksolution(iter, x, resnorm)
        @info("it $(iter): residual norm =  $(resnorm)")
    end
    
    t1 = time()
    Kr_ff = spzeros(size(Phi, 2), size(Phi, 2))
    for partition in partition_list
        Kr_ff += (Phi' * partition.nonoverlapping_K * Phi)        
    end
    Krfactor = lu(Kr_ff)
    @info "Create global factor: $(time() - t1)"
    @info("Global reduced factor: $(mebibytes(Krfactor)) [MiB]")
    
    t0 = time()
    M! = preconditioner!(Krfactor, Phi, partition_list)
    (u_f, stats) = pcg_seq(
        (q, p) -> partition_multiply!(q, partition_list, p), 
        F_f, zeros(size(F_f));
        (M!)=(q, p) -> M!(q, p),
        peeksolution=peeksolution,
        itmax=itmax, 
        # atol=0, rtol=relrestol, normtype = KSP_NORM_NATURAL
        atol= 0, rtol=relrestol, normtype = KSP_NORM_UNPRECONDITIONED
        )
    t1 = time()
    @info("Number of iterations:  $(stats.niter)")
    stats = (niter = stats.niter, residuals = stats.residuals ./ norm(F_f))
    data = Dict(
        "nfreedofs_dchi" => nfreedofs(dchi),
        "ncpartitions" => ncpartitions,
        "nfpartitions" => nfpartitions,
        "overlap" => overlap,
        "size_Kr_ff" => size(Kr_ff),
        "stats" => stats,
        "time" => t1 - t0,
    )
    f = "barrel" *
        "-rf=$(ref)" *
        "-ne=$(nelperpart)" *
        "-n1=$(nbf1max)"  * 
        "-nf=$(nfpartitions)"  * 
        "-ov=$(overlap)"  
    DataDrop.store_json(f * ".json", data)
    scattersysvec!(dchi, u_f)
    
    if visualize
        # f = "barrel" *
        #     "-rf=$(ref)" *
        #     "-ne=$(nelperpart)" *
        #     "-n1=$(nbf1max)" * 
        #     "-nf=$(nfpartitions)"  * 
        #     "-cg-sol"
        # VTK.vtkexportmesh(f * ".vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])
        VTK.vtkexportmesh("barrel-sol.vtk", fens, fes;
            vectors=[("u", deepcopy(dchi.values[:, 1:3]),)])

        p = 1
        for nodelist in nodelists
            cel = connectedelems(fes, nodelist, count(fens))
            vtkexportmesh("barrel-patch$(p).vtk", fens, subset(fes, cel))
            sfes = FESetP1(reshape(nodelist, length(nodelist), 1))
            vtkexportmesh("barrel-nodes$(p).vtk", fens, sfes)
            p += 1
        end
    end

    true
end

function test(;nelperpart = 200, nbf1max = 3, nfpartitions = 8, overlap = 3, ref = 1, stabilize = false, itmax = 2000, relrestol = 1e-6, visualize = false) 
    _execute(32, nelperpart, nbf1max, nfpartitions, overlap, ref, itmax, relrestol, stabilize, visualize)
end

nothing
end # module


