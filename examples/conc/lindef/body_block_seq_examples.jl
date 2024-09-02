module body_block_seq_examples
using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtools.MeshExportModule: CSV
using FinEtools.MeshTetrahedronModule: tetv
using FinEtoolsDeforLinear
using FinEtoolsDDMethods
using FinEtoolsDDMethods: mebibytes
using FinEtoolsDDMethods.CGModule: pcg_seq
using FinEtoolsDDMethods.PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData, npartitions, partition_size 
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

function _execute(label, kind, Ef, nuf, 
    Nepc, nbf1max, nfpartitions, overlap, ref, 
    mesh, boundary_rule, interior_rule, make_femm, itmax, relrestol, visualize)
    CTE = 0.0
    magn = 1.0
    
    fens, fes = mesh(1.0, 1.0, 1.0, 10*ref, 10*ref, 10*ref)

    if visualize
        f = "block-$(label)-$(string(kind))" *
                "-rf=$(ref)" *
                "-mesh"
        vtkexportmesh(f * ".vtk", fens, fes; scalars=[("label", fes.label)])
    end
    
    MR = DeforModelRed3D
    matf = MatDeforElastIso(MR, 0.0, Ef, nuf, CTE)
    
    geom = NodalField(fens.xyz)
    u = NodalField(zeros(size(fens.xyz, 1), 3)) # displacement field

    # Renumber the nodes
    femm = FEMMBase(IntegDomain(fes, interior_rule))
    C = connectionmatrix(femm, count(fens))
    perm = symrcm(C)
    
    bfes = meshboundary(fes)
    lx0 = connectednodes(bfes)
    setebc!(u, lx0, true, 1, 0.0)
    setebc!(u, lx0, true, 2, 0.0)
    setebc!(u, lx0, true, 3, 0.0)
    applyebc!(u)
    numberdofs!(u, perm)
    fr = dofrange(u, DOF_KIND_FREE)
    dr = dofrange(u, DOF_KIND_DATA)
    
    @info("Kind: $(string(kind))")
    @info("Materials: $(Ef), $(nuf)")
    @info("Refinement factor: $(ref)")
    @info("Number of elements per cluster: $(Nepc)")
    @info("Number of 1D basis functions: $(nbf1max)")
    @info("Number of fine grid partitions: $(nfpartitions)")
    @info("Overlap: $(overlap)")
    @info("Number of elements: $(count(fes))")
    @info("Number of free dofs = $(nfreedofs(u))")

    getforce!(forceout, XYZ, tangents, feid, qpid) = (forceout .= [sin(2*pi*XYZ[1]), cos(6*pi*XYZ[2]), -sin(5*pi*XYZ[3])])
    fi = ForceIntensity(Float64, 3, getforce!)
    femm = FEMMBase(IntegDomain(fes, interior_rule))
    F = distribloads(femm, geom, u, fi, 3)
    F_f = F[fr]

    t1 = time()
    cpartitioning, ncpartitions = cluster_partitioning(fens, fes, fes.label, Nepc)
    @info("Number of clusters (coarse grid partitions): $(ncpartitions)")
        
    mor = CoNCData(fens, cpartitioning)
    Phi = transfmatrix(mor, LegendreBasis, nbf1max, u)
    Phi = Phi[fr, :]
    @info("Size of the reduced problem: $(size(Phi, 2))")
    @info("Transformation matrix: $(mebibytes(Phi)) [MiB]")
    @info "Generate clusters ($(round(time() - t1, digits=3)) [s])"

    function make_matrix(_fes)
        _femmf = make_femm(MR, IntegDomain(_fes, interior_rule), matf)
        associategeometry!(_femmf, geom)
        return stiffness(_femmf, geom, u)
    end

    t1 = time()
    cpi = CoNCPartitioningInfo(fens, fes, nfpartitions, overlap, u) 
    partition_list = make_partitions(cpi, fes, make_matrix, nothing)
    @info "Mean fine partition size = $(mean([partition_size(_p) for _p in partition_list]))"
    _b = mean([mebibytes(_p) for _p in partition_list])
    @info "Mean partition allocations: $(Int(round(_b, digits=0))) [MiB]" 
    @info "Total partition allocations: $(sum([mebibytes(_p) for _p in partition_list])) [MiB]" 
    @info "Create partitions time: $(time() - t1)"

    t1 = time()
    Kr_ff = spzeros(size(Phi, 2), size(Phi, 2))
    for partition in partition_list
        Kr_ff += (Phi' * partition.nonoverlapping_K * Phi)
    end
    Krfactor = lu(Kr_ff)
    Kr_ff = nothing
    GC.gc()
    @info "Create global factor: $(time() - t1)"
    @info("Global reduced factor: $(mebibytes(Krfactor)) [MiB]")

    
    function peeksolution(iter, x, resnorm)
        @info("it $(iter): residual norm =  $(resnorm)")
    end
    
    t0 = time()
    M! = preconditioner!(Krfactor, Phi, partition_list)
    (u_f, stats) = pcg_seq(
        (q, p) -> partition_multiply!(q, partition_list, p), 
        F_f, zeros(size(F_f));
        (M!)=(q, p) -> M!(q, p),
        # peeksolution=peeksolution,
        itmax=itmax, 
        # atol=0, rtol=relrestol, normtype = KSP_NORM_NATURAL
        # atol=relrestol * norm(F_f), rtol=0, normtype = KSP_NORM_NATURAL
        atol= 0, rtol=relrestol, normtype = KSP_NORM_UNPRECONDITIONED
        )
    t1 = time()
    @info("Number of iterations:  $(stats.niter)")
    stats = (niter = stats.niter, residuals = stats.residuals ./ norm(F_f))
    data = Dict(
        "nfreedofs_u" => nfreedofs(u),
        "ncpartitions" => ncpartitions,
        "nfpartitions" => nfpartitions,
        "overlap" => overlap,
        "size_Kr_ff" => size(Krfactor),
        "stats" => stats,
        "time" => t1 - t0,
    )
    f = "block-$(label)-$(string(kind))" *
        "-rf=$(ref)" *
        "-ne=$(Nepc)" *
        "-n1=$(nbf1max)"  * 
        "-nf=$(nfpartitions)"  * 
        "-ov=$(overlap)"  
    DataDrop.store_json(f * ".json", data)
    scattersysvec!(u, u_f)

    if visualize
        f = "block-$(label)-$(string(kind))" *
            "-rf=$(ref)" *
            "-ne=$(Nepc)" *
            "-n1=$(nbf1max)" *
            "-nf=$(nfpartitions)" *
            "-cg-sol"
        VTK.vtkexportmesh(f * ".vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])
        
    end
    
    true
end
# test(ref = 1)

function test(label = "soft_hard"; kind = "hex", Ef = 1.0e5, nuf = 0.3, Nepc = 200, nbf1max = 5, nfpartitions = 2, overlap = 1, ref = 1, itmax = 2000, relrestol = 1e-6, visualize = false)
    
    mesh, boundary_rule, interior_rule, make_femm = if kind == "hex"
        mesh = H8block
        boundary_rule = GaussRule(2, 2)
        interior_rule = GaussRule(3, 2)
        make_femm = FEMMDeforLinearMSH8
        (mesh, boundary_rule, interior_rule, make_femm)
    else
        mesh = T10block
        boundary_rule = TriRule(6)
        interior_rule = TetRule(4)
        make_femm = FEMMDeforLinearMST10
        (mesh, boundary_rule, interior_rule, make_femm)
    end
    _execute(label, kind, Ef, nuf, Nepc, nbf1max, nfpartitions, overlap, ref, mesh, boundary_rule, interior_rule, make_femm, itmax, relrestol, visualize)
end

nothing
end # module


