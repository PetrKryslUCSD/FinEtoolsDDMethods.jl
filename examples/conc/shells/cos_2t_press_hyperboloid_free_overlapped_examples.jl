module cos_2t_press_hyperboloid_free_overlapped_examples
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

function _execute(ncoarse, aspect, nelperpart, nbf1max, nfpartitions, overlap, ref, itmax, relrestol, visualize)
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
    
    println("Refinement factor: $(ref)")
    println("Number of elements per partition: $(nelperpart)")
    println("Number of 1D basis functions: $(nbf1max)")
    println("Number of fine grid partitions: $(nfpartitions)")
    println("Overlap: $(overlap)")
    println("Number of elements: $(count(fes))")
    println("Number of free dofs = $(nfreedofs(dchi))")

    lfemm = FEMMBase(IntegDomain(fes, TriRule(3)))
    fi = ForceIntensity(Float64, 6, computetrac!);
    F = distribloads(lfemm, geom0, dchi, fi, 2);
    F_f = F[fr]

    associategeometry!(femm, geom0)
    K = stiffness(femm, geom0, u0, Rfield0, dchi);
    K_ff = K[fr, fr]

    # U_f = K_ff \ F_f
    # scattersysvec!(u, U_f)

    # VTK.vtkexportmesh("fibers-tet-sol.vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])   

    cpartitioning, ncpartitions = shell_cluster_partitioning(fens, fes, nelperpart)
    println("Number of clusters (coarse grid partitions): $(ncpartitions)")
        
    if visualize
        partitionsfes = FESetP1(reshape(1:count(fens), count(fens), 1))
        vtkexportmesh("cos_2t_press_hyperboloid_free-clusters.vtk", fens, partitionsfes; scalars=[("partition", cpartitioning)])
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
    nodelists = CompatibilityModule.fine_grid_node_lists(fens, fes, nfpartitions, overlap)
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

    t0 = time()
    norm_F_f = norm(F_f)
    (u_f, stats) = pcg_seq((q, p) -> mul!(q, K_ff, p), F_f, zeros(size(F_f));
        (M!)=(q, p) -> M!(q, p),
        itmax=itmax, atol=relrestol * norm_F_f, rtol=0)
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
    f = "cos_2t_press_hyperboloid_free_overlapped" *
        "-as=$(aspect)" *
        "-rf=$(ref)" *
        "-ne=$(nelperpart)" *
        "-n1=$(nbf1max)"  * 
        "-nf=$(nfpartitions)"  * 
        "-ov=$(overlap)"  
    DataDrop.store_json(f * ".json", data)
    scattersysvec!(dchi, u_f)
    
    if visualize
        VTK.vtkexportmesh("cos_2t_press_hyperboloid_free-sol.vtk", fens, fes;
            vectors=[("u", deepcopy(dchi.values[:, 1:3]),)])

        p = 1
        for nodelist in nodelists
            cel = connectedelems(fes, nodelist, count(fens))
            vtkexportmesh("cos_2t_press_hyperboloid_free-subdomain-patch$(p).vtk", fens, subset(fes, cel))
            sfes = FESetP1(reshape(nodelist, length(nodelist), 1))
            vtkexportmesh("cos_2t_press_hyperboloid_free-subdomain-nodes$(p).vtk", fens, sfes)
            p += 1
        end
    end

    true
end

function test(;aspect = 100, nelperpart = 100, nbf1max = 5, nfpartitions = 2, overlap = 1, ref = 1, itmax = 2000, relrestol = 1e-6, visualize = false) 
    _execute(32, aspect, nelperpart, nbf1max, nfpartitions, overlap, ref, itmax, relrestol, visualize)
end

nothing
end # module


