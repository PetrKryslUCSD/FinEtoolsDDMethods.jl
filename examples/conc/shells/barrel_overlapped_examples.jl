"""
Free vibration of steel barrel

The structure represents a barrel container with spherical caps, two T
stiffeners, and edge stiffener. This example tests the ability to represent
creases in the surface, and junctions with more than two shell faces joined
along a an intersection.

"""
module barrel_overlapped_examples
using FinEtools
using FinEtools.MeshExportModule: VTK, VTKWrite
using FinEtoolsDeforLinear
using FinEtoolsFlexStructures.FESetShellT3Module: FESetShellT3
using FinEtoolsFlexStructures.FESetShellQ4Module: FESetShellQ4
using FinEtoolsFlexStructures.FEMMShellT3FFModule
using FinEtoolsFlexStructures.RotUtilModule: initial_Rfield, update_rotation_field!
using FinEtoolsDDMethods
using FinEtoolsDDMethods.CGModule: pcg_seq
using FinEtoolsDDMethods.PartitionCoNCDDModule: patch_coordinates, element_overlap
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

function zcant!(csmatout, XYZ, tangents, feid, qpid)
    r = vec(XYZ); 
    cross3!(r, view(tangents, :, 1), view(tangents, :, 2))
    csmatout[:, 3] .= vec(r)/norm(vec(r))
    csmatout[:, 1] .= (1.0, 0.0, 0.0)
    cross3!(view(csmatout, :, 2), view(csmatout, :, 3), view(csmatout, :, 1))
    return csmatout
end

function coarse_grid_partitioning(fens, fes, elem_per_partition = 50, crease_ang = 30/180*pi, cluster_max_normal_deviation = 2 * crease_ang)
    partitioning = zeros(Int, count(fens))
    surfids, partitionids, surface_elem_per_partition = create_partitions(fens, fes, elem_per_partition;
        crease_ang=crease_ang, cluster_max_normal_deviation=cluster_max_normal_deviation)
    for i in eachindex(fes)
        for k in fes.conn[i]
            partitioning[k] = partitionids[i]
        end
    end
    npartitions = maximum(partitioning)
    partitioning, npartitions
end

function make_overlapping_partition(fens, fes, n2e, overlap, element_1st_partitioning, i)
    enl = findall(x -> x == i, element_1st_partitioning)
    touched = fill(false, count(fes)) 
    touched[enl] .= true 
    for ov in 1:overlap
        sfes = subset(fes, enl)
        bsfes = meshboundary(sfes)
        addenl = Int[]
        for e in eachindex(bsfes)
            for n in bsfes.conn[e]
                for ne in n2e.map[n]
                    if !touched[ne] 
                        touched[ne] = true
                        push!(addenl, ne)
                    end
                end
            end
        end
        enl = cat(enl, addenl; dims=1)
    end
    # vtkexportmesh("i=$i" * "-enl" * "-final" * ".vtk", fens, subset(fes, enl))
    return connectednodes(subset(fes, enl))
end

function fine_grid_node_lists(fens, fes, npartitions, overlap)
    femm = FEMMBase(IntegDomain(fes, PointRule()))
    C = dualconnectionmatrix(femm, fens, nodesperelem(boundaryfe(fes)))
    g = Metis.graph(C; check_hermitian=true)
    element_1st_partitioning = Metis.partition(g, npartitions; alg=:KWAY)
    npartitions = maximum(element_1st_partitioning)
    n2e = FENodeToFEMap(fes, count(fens))
    nodelists = []
    for i in 1:npartitions
        push!(nodelists, make_overlapping_partition(fens, fes, n2e, overlap, element_1st_partitioning, i))
    end
    nodelists
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
    
    println("Refinement factor: $(ref)")
    println("Number of elements per partition: $(nelperpart)")
    println("Number of 1D basis functions: $(nbf1max)")
    println("Number of fine grid partitions: $(nfpartitions)")
    println("Overlap: $(overlap)")
    println("Number of elements: $(count(fes))")
    println("Number of free dofs = $(nfreedofs(dchi))")

    lfemm = FEMMBase(IntegDomain(subset(fes, vessel), TriRule(3)))
    fi = ForceIntensity(Float64, 6, computetrac!);
    F = distribloads(lfemm, geom0, dchi, fi, 2);
    F_f = F[fr]

    associategeometry!(femm, geom0)
    K = stiffness(femm, geom0, u0, Rfield0, dchi);
    K_ff = K[fr, fr]

    # scattersysvec!(dchi, F, DOF_KIND_ALL)
    # VTK.vtkexportmesh("forces.vtk", fens, fes; vectors=[("F", deepcopy(dchi.values[:, 1:3]),)])   
    # VTK.vtkexportmesh("fibers-tet-sol.vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])   

    cpartitioning, ncpartitions = coarse_grid_partitioning(fens, fes, nelperpart)
    println("Number coarse grid partitions: $(ncpartitions)")
        
    if visualize
        f = "barrel_overlapped" *
            "-rf=$(ref)" *
            "-ne=$(nelperpart)" * "-partitioning"
        partitionsfes = FESetP1(reshape(1:count(fens), count(fens), 1))
        vtkexportmesh(f * ".vtk", fens, partitionsfes; scalars=[("partition", cpartitioning)])
        # @async run(`"paraview.exe" $File`)
    end
       
    
    mor = CoNCData(list -> patch_coordinates(fens.xyz, list), cpartitioning)
    Phi = transfmatrix(mor, LegendreBasis, nbf1max, dchi)
    Phi = Phi[fr, :]
    transfm(m, t, tT) = (tT * m * t)
    transfv(v, t, tT) = (tT * v)
    PhiT = Phi'
    Kr_ff = transfm(K_ff, Phi, PhiT)
    println("Size of the reduced problem: $(size(Kr_ff))")
    Krfactor = lu(Kr_ff)

    # display(MatrixSpy.spy_matrix(sparse(Phi'), "Phi"))

    # U_f = Phi * (Krfactor \ (PhiT * F_f))
    # scattersysvec!(u, U_f)

    # VTK.vtkexportmesh("fibers-tet-red-sol.vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])   
    
    # fpartitioning, nfpartitions = fine_grid_partitioning(fens, nfpartitions)
    nodelists = fine_grid_node_lists(fens, fes, nfpartitions, overlap)
    @assert length(nodelists) == nfpartitions
    
    partitions = []
    for nodelist in nodelists
        doflist = Int[]
        for n in nodelist
            for d in axes(dchi.dofnums, 2)
                if dchi.dofnums[n, d] in fr
                    push!(doflist, dchi.dofnums[n, d])
                end
            end
        end
        pK = K[doflist, doflist]
        pKfactor = lu(pK)
        part = (nodelist = nodelist, factor = pKfactor, doflist = doflist)
        push!(partitions, part)
    end

    meansize = mean([length(part.doflist) for part in partitions])
    println("Mean fine partition size: $(meansize)")

    function M!(q, p)
        q .= Phi * (Krfactor \ (PhiT * p))
        for part in partitions
            q[part.doflist] .+= (part.factor \ p[part.doflist])
        end
        q
    end

    # peeksolution(iter, x) = begin
    #     println("Iteration: $(iter)")
    #     if iter % 10 == 0 
    #         if iter > 10
    #             f = "barrel_overlapped" *
    #                 "-rf=$(ref)" *
    #                 "-ne=$(nelperpart)" *
    #                 "-n1=$(nbf1max)" *
    #                 "-nf=$(nfpartitions)" *
    #                 "-ov=$(overlap)" *
    #                 "-cg-sol-iter$(iter-10)"
    #             xp = DataDrop.retrieve_matrix(f * ".h5", "x")
    #             scattersysvec!(dchi, x - xp)
    #             VTK.vtkexportmesh("barrel-xincr-iter=$(iter).vtk", fens, fes;
    #                 vectors=[("u", deepcopy(dchi.values[:, 1:3]),)])
    #         end
    #         f = "barrel_overlapped" *
    #             "-rf=$(ref)" *
    #             "-ne=$(nelperpart)" *
    #             "-n1=$(nbf1max)" *
    #             "-nf=$(nfpartitions)" *
    #             "-ov=$(overlap)" * 
    #             "-cg-sol-iter$(iter)"
    #         DataDrop.empty_hdf5_file(f * ".h5")
    #         DataDrop.store_matrix(f * ".h5", "x", x)
    #     end
    # end

    t0 = time()
    norm_F_f = norm(F_f)
    (u_f, stats) = pcg_seq((q, p) -> mul!(q, K_ff, p), F_f, zeros(size(F_f));
        (M!)=(q, p) -> M!(q, p),
        itmax=itmax, atol= relrestol * norm_F_f, rtol=0)
    t1 = time()
    println("Number of iterations:  $(stats.niter)")
    stats = (niter = stats.niter, residuals = stats.residuals ./ norm_F_f)
    data = Dict(
        "nfreedofs_dchi" => nfreedofs(dchi),
        "ncpartitions" => ncpartitions,
        "nfpartitions" => nfpartitions,
        "overlap" => overlap,
        "size_Kr_ff" => size(Kr_ff),
        "stats" => stats,
        "time" => t1 - t0,
    )
    f = "barrel_overlapped" *
        "-rf=$(ref)" *
        "-ne=$(nelperpart)" *
        "-n1=$(nbf1max)"  * 
        "-nf=$(nfpartitions)"  * 
        "-ov=$(overlap)"  
    DataDrop.store_json(f * ".json", data)
    scattersysvec!(dchi, u_f)
    
    if visualize
        # f = "barrel_overlapped" *
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


