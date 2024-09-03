module hyp_free_seq_examples

using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsDeforLinear
using FinEtoolsFlexStructures.FESetShellT3Module: FESetShellT3
using FinEtoolsFlexStructures.FESetShellQ4Module: FESetShellQ4
using FinEtoolsFlexStructures.FEMMShellT3FFModule
using FinEtoolsFlexStructures.RotUtilModule: initial_Rfield, update_rotation_field!
using FinEtoolsDDMethods
using FinEtoolsDDMethods.CGModule: pcg_seq
using FinEtoolsDDMethods.CoNCUtilitiesModule: patch_coordinates
using FinEtoolsDDMethods.CompatibilityModule
using FinEtoolsDDMethods.PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData, npartitions 
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
using Statistics
# using MatrixSpy

# Parameters:
E = 2.0e11
nu = 1/3;
pressure = 1.0e3;
Length = 2.0;

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

function _execute(filename, ncoarse, ref, aspect, Nc, n1, Np, No, itmax, relrestol, peek, visualize)
    CTE = 0.0
    distortion = 0.0
    n = ncoarse  * ref    # number of elements along the edge of the block
    tolerance = Length / n/ 100
    thickness = Length/2/aspect
    
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
    
    @info "Aspect ratio: $(aspect)"
    @info("Refinement factor: $(ref)")
    @info("Number of fine grid partitions: $(Np)")
    @info("Number of overlaps: $(No)")
    @info("Number of elements: $(count(fes))")
    @info("Number of free dofs = $(nfreedofs(dchi))")

    lfemm = FEMMBase(IntegDomain(fes, TriRule(3)))
    fi = ForceIntensity(Float64, 6, computetrac!);
    F = distribloads(lfemm, geom0, dchi, fi, 2);
    F_f = F[fr]

    associategeometry!(femm, geom0)
    
    function make_matrix(fes)
        femm.integdomain.fes = fes
        return stiffness(femm, geom0, u0, Rfield0, dchi);
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
    n1adj = n1 + 1 # adjust for a safety margin
    ntadj = 0.8 * n1*(n1+1)/2 + 0.2 * n1adj*(n1adj+1)/2
    (Nc == 0) && (Nc = Int(floor(minimum(partition_sizes) / ntadj / ndofs(dchi))))
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
        "nfreedofs" => nfreedofs(dchi),
        "Nc" => Nc,
        "Np" => Np,
        "No" => No,
        "size_Kr_ff" => size(Krfactor),
        "stats" => stats,
        "time" => t1 - t0,
    )
    f = (filename == "" ?
         "hyp_free" *
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
         "hyp_free-" *
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

function test(;filename = "", ref = 2, aspect = 10.0, Nc = 2, n1 = 6, Np = 2, No = 1, itmax = 2000, relrestol = 1e-6, peek = false, visualize = false) 
    ncoarse = 32
    _execute(filename, ncoarse, ref, aspect, Nc, n1, Np, No, itmax, relrestol, peek, visualize)
end

nothing
end # module


