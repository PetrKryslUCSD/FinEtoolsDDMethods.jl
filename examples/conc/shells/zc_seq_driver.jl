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
module zc_seq_examples

using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsDeforLinear
using FinEtoolsFlexStructures.FESetShellT3Module: FESetShellT3
using FinEtoolsFlexStructures.FESetShellQ4Module: FESetShellQ4
using FinEtoolsFlexStructures.FEMMShellT3FFModule
using FinEtoolsFlexStructures.RotUtilModule: initial_Rfield, update_rotation_field!
using FinEtoolsDDMethods
using FinEtoolsDDMethods.CGModule: pcg_seq, vec_copyto!
using FinEtoolsDDMethods.CoNCUtilitiesModule: patch_coordinates
using FinEtoolsDDMethods.PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData, npartitions 
using FinEtoolsDDMethods.DDCoNCSeqModule: make_partitions, PartitionedVector, aop!, preconditioner!, vec_copyto!
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

function _execute(filename, ref, Nc, n1, Np, No, itmax, relrestol, peek, visualize)
    CTE = 0.0
    thickness = 0.1
    
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
    
    @info("Refinement factor: $(ref)")
    @info("Number of fine grid partitions: $(Np)")
    @info("Number of overlaps: $(No)")
    @info("Number of elements: $(count(fes))")
    @info("Number of free dofs = $(nfreedofs(dchi))")

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
    
    function make_matrix(fes)
        femm1 = deepcopy(femm) # for thread safety
        femm1.integdomain.fes = fes
        return stiffness(femm1, geom0, u0, Rfield0, dchi);
    end

    t1 = time()
    cpi = CoNCPartitioningInfo(fens, fes, Np, No, dchi) 
    partition_list  = make_partitions(cpi, fes, make_matrix, nothing)
    partition_sizes = [partition_size(_p) for _p in partition_list]
    meanps = mean(partition_sizes)
    @info "Mean fine partition size: $(meanps)"
    @info "Create partitions ($(round(time() - t1, digits=3)) [s])"

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
    @info "Generate clusters ($(round(time() - t1, digits=3)) [s])"

    function peeksolution(iter, x, resnorm)
        peek && (@info("it $(iter): residual norm =  $(resnorm)"))
    end
    
    t1 = time()
    Kr_ff = spzeros(size(Phi, 2), size(Phi, 2))
    for partition in partition_list
        Kr_ff += (Phi' * partition.nonoverlapping_K * Phi)
    end
    Krfactor = lu(Kr_ff)
    @info "Create global factor ($(round(time() - t1, digits=3)) [s])"
    
    t0 = time()
    M! = preconditioner!(Krfactor, Phi, partition_list)
    (u_f, stats) = pcg_seq(
        (q, p) -> partition_multiply!(q, partition_list, p), 
        F_f, zeros(size(F_f));
        (M!)=(q, p) -> M!(q, p),
        peeksolution=peeksolution,
        itmax=itmax, 
        # atol=0, rtol=relrestol, normtype = KSP_NORM_NATURAL
        # atol=relrestol * norm(F_f), rtol=0, normtype = KSP_NORM_NATURAL
        atol= 0, rtol=relrestol, normtype = KSP_NORM_UNPRECONDITIONED
        )
    t1 = time()
    @info("Number of iterations:  $(stats.niter)")
    @info "Iterations ($(round(t1 - t0, digits=3)) [s])"
    stats = (niter = stats.niter, residuals = stats.residuals ./ norm(F_f))
    data = Dict(
        "number_nodes" => count(fens),
        "number_elements" => count(fes),
        "nfreedofs" => nfreedofs(dchi),
        "Nc" => Nc,
        "Np" => Np,
        "No" => No,
        "meanps" => meanps,
        "size_Kr_ff" => size(Krfactor),
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
    @info "Storing data in $(f * ".json")"
    DataDrop.store_json(f * ".json", data)
    scattersysvec!(dchi, u_f)
    
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

    true
end

function _execute_alt(filename, ref, Nc, n1, Np, No, itmax, relrestol, peek, visualize)
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
    el = selectelem(fens, bfes, box = Float64[10.0 10.0 1.0 1.0 -1 1], tolerance = tolerance)
    lfemm1 = FEMMBase(IntegDomain(subset(bfes, el), GaussRule(1, 2)))
    fi1 = ForceIntensity(Float64[0, 0, +0.6e6, 0, 0, 0]);
    el = selectelem(fens, bfes, box = Float64[10.0 10.0 -1.0 -1.0 -1 1], tolerance = tolerance)
    lfemm2 = FEMMBase(IntegDomain(subset(bfes, el), GaussRule(1, 2)))
    fi2 = ForceIntensity(Float64[0, 0, -0.6e6, 0, 0, 0]);
    F = distribloads(lfemm1, geom0, dchi, fi1, 1) + distribloads(lfemm2, geom0, dchi, fi2, 1);
    F_f = F[fr]

    associategeometry!(femm, geom0)

    K = stiffness(femm, geom0, u0, Rfield0, dchi)
    K_ff = K[fr, fr]

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
    cpi = CoNCPartitioningInfo(fens, fes, Np, No, dchi; visualize = true) 
    @info "Create CoNCPartitioningInfo ($(round(time() - t1, digits=3)) [s])"
    t2 = time()
    partition_list  = make_partitions(cpi, fes, make_matrix, nothing)
    @info "make_partitions ($(round(time() - t2, digits=3)) [s])"
    partition_sizes = [partition_size(_p) for _p in partition_list]
    meanps = mean(partition_sizes)
    @info "Mean fine partition size: $(meanps)"
    @info "Create partitions ($(round(time() - t1, digits=3)) [s])"

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
    @info "Generate clusters ($(round(time() - t1, digits=3)) [s])"

    function peeksolution(iter, x, resnorm)
        peek && (@info("it $(iter): residual norm =  $(resnorm)"))
    end
    
    K_ff_2 = spzeros(size(K_ff, 1), size(K_ff, 1))
    for partition in partition_list
        d = partition.entity_list.nonshared.global_dofs
        K_ff_2[d, d] += partition.nonshared_K
    end
    # @show norm(K_ff_2)
    # for i in axes(K_ff, 1)
    #     for j in axes(K_ff, 1)
    #         if abs(K_ff[i, j] - K_ff_2[i, j]) / abs(K_ff[i, j]) > 1e-6
    #             @show i, j, K_ff[i, j], K_ff_2[i, j]
    #         end
    #     end
    # end
    @show norm(K_ff - K_ff_2) / norm(K_ff)
    
    t1 = time()
    Kr_ff = spzeros(size(Phi, 2), size(Phi, 2))
    for partition in partition_list
        d = partition.entity_list.nonshared.global_dofs
        P = Phi[d, :]
        Kr_ff += (P' * partition.nonshared_K * P)
    end
    Krfactor = lu(Kr_ff)
    @info "Create global factor ($(round(time() - t1, digits=3)) [s])"
    
    t0 = time()
    x0 = PartitionedVector(Float64, partition_list)
    # Test 1
    vec_copyto!(x0, 0.0)
    q = rand(Float64, size(F_f))
    vec_copyto!(q, x0)
    @show norm(q) == 0
    # Test 2
    q = rand(Float64, size(F_f))
    vec_copyto!(x0, q)
    q1 = rand(Float64, size(F_f))
    vec_copyto!(q1, x0)
    @show norm(q - q1) == 0
    b = PartitionedVector(Float64, partition_list)
    vec_copyto!(b, F_f)
    p = rand(Float64, size(F_f))
    q = rand(Float64, size(F_f))
    vec_copyto!(x0, p)
    q .= 0.0
    vec_copyto!(q, x0)
    @show norm(p - q)
    # Test 3
    q = rand(Float64, size(F_f))
    p = rand(Float64, size(F_f))
    q .= 0.0; p .= 1
    vec_copyto!(x0, p)
    p1 = rand(Float64, size(F_f))
    vec_copyto!(p1, x0)
    @show norm(p - p1)
    # Test aop!(q, p)
    q = rand(Float64, size(F_f))
    p = rand(Float64, size(F_f))
    q .= 0.0; p .= 1
    x0 = PartitionedVector(Float64, partition_list)
    vec_copyto!(x0, p)
    b = PartitionedVector(Float64, partition_list)
    aop!(b, x0)
    vec_copyto!(q, b)
    q1 = K_ff * p
    @show norm(q - q1), norm(q), norm(q1)
    return
    # M! = preconditioner!(Krfactor, Phi, partition_list)
    # q = zeros(size(F_f))
    # a_mult!(q, aop, p)
    # q1 = K_ff * p
    # @show norm(q - q1)
    x0 = PartitionedVector(Float64, partition_list)
    vec_copyto!(x0, 0.0)
    b = PartitionedVector(Float64, partition_list)
    vec_copyto!(b, F_f)
    (u_f, stats) = pcg_seq(
        (q, p) -> aop!(q, p), 
        b, x0;
        (M!)=(q, p) -> vec_copyto!(q, p),
        peeksolution=peeksolution,
        itmax=itmax, 
        # atol=0, rtol=relrestol, normtype = KSP_NORM_NATURAL
        # atol=relrestol * norm(F_f), rtol=0, normtype = KSP_NORM_NATURAL
        atol= 0, rtol=relrestol, normtype = KSP_NORM_UNPRECONDITIONED
        )
    t1 = time()
    @info("Number of iterations:  $(stats.niter)")
    @info "Iterations ($(round(t1 - t0, digits=3)) [s])"
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
        "size_Kr_ff" => size(Krfactor),
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
    @info "Storing data in $(f * ".json")"
    DataDrop.store_json(f * ".json", data)
    # scattersysvec!(dchi, u_f)
    
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
        default = 1
        "--No"
        help = "Number of overlaps"
        arg_type = Int
        default = 1
        "--Np"
        help = "Number of partitions"
        arg_type = Int
        default = 2
        "--ref"
        help = "Refinement factor"
        arg_type = Int
        default = 6
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


