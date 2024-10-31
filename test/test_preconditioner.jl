module test_preconditioner

using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsDeforLinear
using FinEtoolsFlexStructures.FESetShellT3Module: FESetShellT3
using FinEtoolsFlexStructures.FESetShellQ4Module: FESetShellQ4
using FinEtoolsFlexStructures.FEMMShellT3FFModule
using FinEtoolsFlexStructures.RotUtilModule: initial_Rfield, update_rotation_field!
using FinEtoolsDDMethods
using FinEtoolsDDMethods.CGModule: pcg, vec_copyto!
using FinEtoolsDDMethods.CoNCUtilitiesModule: patch_coordinates
using FinEtoolsDDMethods.PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData, npartitions
using FinEtoolsDDMethods.DDCoNCSeqModule: make_partitions, PartitionedVector, aop!, TwoLevelPreConditioner, vec_copyto!
using FinEtoolsDDMethods: set_up_timers
using SymRCM
using Metis
using Test
using LinearAlgebra
using SparseArrays
import Base: size, eltype
import LinearAlgebra: mul!
import CoNCMOR: CoNCData, transfmatrix, LegendreBasis
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

    #@info("Refinement factor: $(ref)")
    #@info("Number of fine grid partitions: $(Np)")
    #@info("Number of overlaps: $(No)")
    #@info("Number of nodes: $(count(fens))")
    #@info("Number of elements: $(count(fes))")
    #@info("Number of free dofs = $(nfreedofs(dchi))")

    function make_matrix(fes)
        femm1 = deepcopy(femm) # for thread safety
        femm1.integdomain.fes = fes
        return stiffness(femm1, geom0, u0, Rfield0, dchi);
    end

    t1 = time()
    cpi = CoNCPartitioningInfo(fens, fes, Np, No, dchi) 
    #@info("Create partitioning info ($(round(time() - t1, digits=3)) [s])")
    t2 = time()
    partition_list  = make_partitions(cpi, fes, make_matrix, nothing)
    #@info "make_partitions ($(round(time() - t2, digits=3)) [s])"
    partition_sizes = [partition_size(_p) for _p in partition_list]
    meanps = mean(partition_sizes)
    #@info("Mean fine partition size: $(meanps)")
    #@info("Create partitions ($(round(time() - t1, digits=3)) [s])")

    t1 = time()
    #@info("Number of clusters (requested): $(Nc)")
    #@info("Number of 1D basis functions: $(n1)")
    nt = n1*(n1+1)/2 
    (Nc == 0) && (Nc = Int(floor(meanps / nt / ndofs(dchi))))
    Nepc = count(fes) ÷ Nc
    (n1 > (Nepc/2)^(1/2)) && @error "Not enough elements per cluster"
    #@info("Number of elements per cluster: $(Nepc)")
    
    cpartitioning, Nc = shell_cluster_partitioning(fens, fes, Nepc)
    #@info("Number of clusters (actual): $(Nc)")
        
    mor = CoNCData(list -> patch_coordinates(fens.xyz, list), cpartitioning)
    Phi = transfmatrix(mor, LegendreBasis, n1, dchi)
    Phi = Phi[fr, :]
    #@info("Size of the reduced problem: $(size(Phi, 2))")
    #@info("Generate clusters ($(round(time() - t1, digits=3)) [s])")

    function peeksolution(iter, x, resnorm)
        peek && (@info("it $(iter): residual norm =  $(resnorm)"))
    end
    
    K_ff_2 = spzeros(size(K_ff, 1), size(K_ff, 1))
    for partition in partition_list
        d = partition.entity_list.nonshared.global_dofs
        K_ff_2[d, d] += partition.Kns_ff
    end
    # @show norm(K_ff_2)
    # for i in axes(K_ff, 1)
    #     for j in axes(K_ff, 1)
    #         if abs(K_ff[i, j] - K_ff_2[i, j]) / abs(K_ff[i, j]) > 1e-6
    #             @show i, j, K_ff[i, j], K_ff_2[i, j]
    #         end
    #     end
    # end
    @test norm(K_ff - K_ff_2) / norm(K_ff) < 1e-10
    
    t1 = time()
    Kr_ff = spzeros(size(Phi, 2), size(Phi, 2))
    for partition in partition_list
        d = partition.entity_list.nonshared.global_dofs
        P = Phi[d, :]
        Kr_ff += (P' * partition.Kns_ff * P)
    end
    Krfactor = lu(Kr_ff)
    #@info "Create global factor ($(round(time() - t1, digits=3)) [s])"
    
    t0 = time()
    M! = TwoLevelPreConditioner(partition_list, Phi)
    #@info("Create preconditioner ($(round(time() - t1, digits=3)) [s])")

    p = 1000 * rand(Float64, size(Krfactor, 1))
    q = Krfactor \ p
    q1 = M!.Kr_ff_factor \ p
    @test norm(q - q1) /  norm(q1) < 1e-9

    b = PartitionedVector(Float64, partition_list)
    p = 1000 * rand(Float64, size(F_f))
    vec_copyto!(b, p)
    r = PartitionedVector(Float64, partition_list)
    M!(r, b)
    q = rand(Float64, size(F_f))
    vec_copyto!(q, r)
    # @show q1 = Phi' * p
    # @show M!.buffPp
    # @test norm(M!.buffPp - q1) / norm(q1) < 1e-9
    q1 = Phi * (Krfactor \ (Phi' * p))
    @test norm(q - q1) / norm(q)  <  1e-9

    true
end

filename = "test_partitioned_vector"
ref = 11
Nc = 5
n1 = 4
Np = 5
No = 1
itmax = 20
relrestol = 1e-6
peek = true
visualize = false

for ref in [13, 11, 15]
    for Np in [11, 1, 2, 16, 9]
        for No in [3, 1, 2]
            for n1 in [1, 4, 6]
                _execute_alt(
                    filename,
                    ref,
                    Nc, 
                    n1,
                    Np, 
                    No, 
                    itmax, 
                    relrestol,
                    peek,
                    visualize
                    )
            end
        end
    end
end

nothing
end # module

